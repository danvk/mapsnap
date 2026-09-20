#!/bin/bash
# Sample the Batch fleet while a job runs, so its cost can be worked out later.
#
#   scripts/batch/snapshot-instances.sh <array-job-id> [out.tsv]
#
# This has to run alongside the job, not after it: a terminated instance
# leaves describe-instances within the hour and takes its spot-reclamation
# reason with it, and Cost Explorer lags a day and cannot tell this job from
# anything else in the account. Each row is one instance at one moment;
# collect-run.py turns them into instance-hours per shape.
#
# It stops on its own when the array reaches SUCCEEDED or FAILED. Run it in
# the background beside status.sh:
#
#   scripts/batch/snapshot-instances.sh "$JOB" instances.tsv &
#   scripts/batch/status.sh "$JOB" --watch
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
JOB=${1:?array job id}; OUT=${2:-instances.tsv}
PERIOD=${PERIOD:-120}

[ -s "$OUT" ] || printf 'ts\tinstance\ttype\taz\tstate\tlaunch\ttransition\n' > "$OUT"
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:project,Values=mapsnap-craft" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime,StateTransitionReason]' \
    --output text 2>/dev/null |
    while IFS=$'\t' read -r id kind az state launch reason; do
      [ -n "${id:-}" ] || continue
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$id" "$kind" "$az" "$state" "$launch" "$reason" >> "$OUT"
    done
  status=$(aws batch describe-jobs --region "$REGION" --jobs "$JOB" \
    --query 'jobs[0].status' --output text 2>/dev/null || echo UNKNOWN)
  case "$status" in
    SUCCEEDED|FAILED)
      # One last look: the scale-down happens after the array finishes, and
      # those minutes are still billed.
      sleep "$PERIOD"
      now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:project,Values=mapsnap-craft" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime,StateTransitionReason]' \
        --output text 2>/dev/null |
        while IFS=$'\t' read -r id kind az state launch reason; do
          [ -n "${id:-}" ] || continue
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$id" "$kind" "$az" "$state" "$launch" "$reason" >> "$OUT"
        done
      echo "$JOB is $status; $(( $(grep -c . "$OUT") - 1 )) snapshots in $OUT" >&2
      exit 0 ;;
  esac
  sleep "$PERIOD"
done
