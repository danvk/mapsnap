#!/bin/bash
# Keep a sharded fleet alive: relaunch any shard that is neither finished nor
# running. Spot instances are reclaimed -- 29 of 30 key-map instances went in one
# second on 2026-09-14 -- and a one-time request leaves nothing behind, so a job
# started at night is simply dead by morning unless something restarts it.
#
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --workers 2 --on-demand-from 2
#   scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --watch      # loop every 15 min
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
while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    # Anything launch.sh understands is handed straight through.
    --workers) WORKERS="$2"; PASSTHROUGH+=(--workers "$2"); shift 2 ;;
    --on-demand-from|--instance-type|--extra-args|--git-ref|--region)
      PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
export AWS_REGION=$REGION
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)

# Shards that have written their completion marker.
finished_shards() {
  aws s3 ls "$BUCKET/_craft/done/" 2>/dev/null \
    | awk '{print $4}' \
    | sed -n "s/^${JOB}-of-${SHARDS}-shard-\([0-9]*\)$/\1/p"
}

# Shards with an instance up right now. Matches on the job tag, and falls back
# to the Name for the first corpus fleet, which launched before --job existed
# and is named mapsnap-craft-<shard> with no job tag at all.
running_shards() {
  aws ec2 describe-instances \
    --filters Name=tag:project,Values=mapsnap-craft Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].[Tags[?Key==`shard`].Value|[0],Tags[?Key==`job`].Value|[0],Tags[?Key==`Name`].Value|[0]]' \
    --output text 2>/dev/null \
  | awk -v job="$JOB" '
      $2 == job { print $1; next }
      $2 == "None" && $3 ~ /^mapsnap-craft-/ && job == "loc-craft" { print $1 }
    ' || true
}

sweep() {
  local finished running relaunched=0 alive=0 done_count=0
  finished=$(finished_shards)
  running=$(running_shards)
  for shard in $(seq 0 $((SHARDS - 1))); do
    if echo "$finished" | grep -qx "$shard"; then
      done_count=$((done_count + 1))
      continue
    fi
    if echo "$running" | grep -qx "$shard"; then
      alive=$((alive + 1))
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
  echo "$(date -u +%H:%M) $JOB: $done_count/$SHARDS finished, $alive running, $relaunched relaunched"
  [ "$done_count" -eq "$SHARDS" ]
}

if [ "$WATCH" -eq 0 ]; then
  sweep && echo "all $SHARDS shards finished"
  exit 0
fi

while true; do
  if sweep; then
    echo "all $SHARDS shards finished"
    exit 0
  fi
  sleep "$INTERVAL"
done
