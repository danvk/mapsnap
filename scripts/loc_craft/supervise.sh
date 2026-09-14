#!/bin/bash
# Keep a sharded fleet alive: relaunch any shard that is neither finished nor
# running. Spot instances are reclaimed -- 29 of 30 key-map instances went in one
# second on 2026-09-14 -- and a one-time request leaves nothing behind, so a job
# started at night is simply dead by morning unless something restarts it.
#
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --workers 2 --on-demand-from 2
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --watch      # loop every 15 min
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --dry-run   # report, launch nothing
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 6 --own 4-5 --region us-east-2
#
# --own names the shards this supervisor is responsible for, which is what a
# fleet split across regions needs: each region runs its own supervisor, and
# without it every one of them would relaunch the other regions' shards into
# its own region, duplicating the work the partition was meant to divide.
#
# Relaunches clone origin/main unless --git-ref names something else.
#
# A shard is finished when it has written _craft/done/<job>-of-<shards>-shard-<n>,
# which bootstrap.sh does only on a clean exit. Finished and reclaimed look
# identical from EC2 (the instance is gone), so the marker is the only way to
# tell them apart, and relaunching a finished shard would be harmless but
# endless.
#
# Safe to run repeatedly: the work itself is idempotent, so a relaunched shard
# skips everything already done and picks up where the dead one stopped.
#
# One-shot by default, so it can live in cron or launchd and survive a laptop
# restart, which a --watch loop in a terminal does not:
#
#   */15 * * * * cd ~/github/mapsnap && scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --workers 2 >> /tmp/supervise.log 2>&1
set -euo pipefail

JOB=loc-craft
SHARDS=4
WORKERS=1
ONLY_ARGS=()
INTERVAL=900
WATCH=0
BUCKET=${BUCKET:-s3://mapsnap-sanborn}
REGION=${AWS_REGION:-us-west-2}
PASSTHROUGH=()
GIT_REF=""
OWN=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;   # report what a sweep would launch
    --interval) INTERVAL="$2"; shift 2 ;;
    # Anything launch.sh understands is handed straight through.
    --workers) WORKERS="$2"; PASSTHROUGH+=(--workers "$2"); shift 2 ;;
    --own) OWN="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --on-demand-from|--instance-type|--extra-args|--region)
      PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
export AWS_REGION=$REGION
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)
source "$HERE/shards.sh"
MINE=$(expand_shards "${OWN:-0-$((SHARDS - 1))}" "$SHARDS")
MINE_COUNT=$(echo "$MINE" | wc -l | tr -d " ")

# launch.sh clones at the working tree's HEAD, so an unattended sweep would send
# the fleet whatever branch happened to be checked out. Pin relaunches to
# origin/main instead, re-resolved each sweep so merged fixes ship on restart.
if [ -z "$GIT_REF" ]; then
  git -C "$HERE" fetch --quiet origin main 2> /dev/null || true
  GIT_REF=$(git -C "$HERE" rev-parse origin/main)
fi
PASSTHROUGH+=(--git-ref "$GIT_REF")

# Shards that have written their completion marker. An empty prefix makes
# `aws s3 ls` exit 1, which is not a failure here, so only a credential-shaped
# error (exit 255) aborts the sweep.
finished_shards() {
  local raw status=0
  raw=$(aws s3 ls "$BUCKET/_craft/done/" 2>/dev/null) || status=$?
  if [ "$status" -ge 2 ]; then
    return 1
  fi
  echo "$raw" \
    | awk '{print $4}' \
    | sed -n "s/^${JOB}-of-${SHARDS}-shard-\([0-9]*\)$/\1/p"
}

# Shards with an instance up right now. Matches on the job tag, and falls back
# to the Name for the first corpus fleet, which launched before --job existed
# and is named mapsnap-craft-<shard> with no job tag at all.
#
# A failed call must NOT be read as "nothing is running": that is how a
# supervisor launches a second instance onto every shard it already has.
# sweep() aborts instead, and the next sweep tries again.
running_shards() {
  local raw
  raw=$(aws ec2 describe-instances \
    --filters Name=tag:project,Values=mapsnap-craft Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].[Tags[?Key==`shard`].Value|[0],Tags[?Key==`job`].Value|[0],Tags[?Key==`Name`].Value|[0]]' \
    --output text) || return 1
  echo "$raw" | awk -v job="$JOB" '
      $2 == job { print $1; next }
      $2 == "None" && $3 ~ /^mapsnap-craft-/ && job == "loc-craft" { print $1 }
    '
}

sweep() {
  local finished running relaunched=0 alive=0 done_count=0
  if ! finished=$(finished_shards); then
    echo "$(date -u +%H:%M) cannot list $BUCKET/_craft/done/; skipping this sweep" >&2
    return 1
  fi
  if ! running=$(running_shards); then
    echo "$(date -u +%H:%M) cannot list instances; skipping this sweep" >&2
    return 1
  fi
  for shard in $MINE; do
    if echo "$finished" | grep -qx "$shard"; then
      done_count=$((done_count + 1))
      continue
    fi
    if echo "$running" | grep -qx "$shard"; then
      alive=$((alive + 1))
      continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "$(date -u +%H:%M) shard $shard: would relaunch (--dry-run)"
      relaunched=$((relaunched + 1))
      continue
    fi
    echo "$(date -u +%H:%M) shard $shard: neither finished nor running, relaunching"
    if "$HERE/launch.sh" --job "$JOB" --shards "$SHARDS" --only "$shard" \
        ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
      relaunched=$((relaunched + 1))
    else
      echo "$(date -u +%H:%M) shard $shard: relaunch failed (quota or capacity); will retry" >&2
    fi
  done
  echo "$(date -u +%H:%M) $JOB in $REGION: $done_count/$MINE_COUNT finished, $alive running, $relaunched relaunched"
  [ "$done_count" -eq "$MINE_COUNT" ]
}

if [ "$WATCH" -eq 0 ]; then
  sweep && echo "all $MINE_COUNT owned shards finished"
  exit 0
fi

while true; do
  if sweep; then
    echo "all $MINE_COUNT owned shards finished"
    exit 0
  fi
  sleep "$INTERVAL"
done
