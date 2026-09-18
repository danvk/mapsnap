#!/bin/bash
# The status page Batch's console does not quite give you: an array job's
# children by state, throughput and ETA from the finished ones, and every
# failed child with its exit code and reason, on one screen.
#
#   scripts/batch/status.sh <array-job-id>
#   scripts/batch/status.sh <array-job-id> --watch     # every 60 s until done
#
# The console (Jobs -> the array job) shows the same state counts and, per
# child, its status, exit code, status reason and a link to its CloudWatch
# log stream; it does not compute a rate or an ETA, and the failed children
# are a filter away. This prints them. Exit codes are loc-fit's:
#   0 fitted / already done   1 the chain raised (retried once)
#   3 unprocessable           4 inputs missing            137 killed (memory)
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
JOB=${1:?array job id}; WATCH=${2:-}

show() {
  local j; j=$(aws batch describe-jobs --region "$REGION" --jobs "$JOB" --query 'jobs[0]' --output json)
  local name status created size
  name=$(jq -r .jobName <<< "$j"); status=$(jq -r .status <<< "$j")
  created=$(jq -r '.createdAt / 1000 | floor' <<< "$j"); size=$(jq -r '.arrayProperties.size // 1' <<< "$j")
  local now elapsed; now=$(date +%s); elapsed=$(( now - created ))
  echo "$name  ($JOB)  $status  size $size  elapsed $(( elapsed / 60 ))m"
  jq -r '.arrayProperties.statusSummary // {} | to_entries | map("\(.key) \(.value)") | join("  ")' <<< "$j"
  local ok fail; ok=$(jq -r '.arrayProperties.statusSummary.SUCCEEDED // 0' <<< "$j"); fail=$(jq -r '.arrayProperties.statusSummary.FAILED // 0' <<< "$j")
  if [ "$ok" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
    local rate left; rate=$(python3 -c "print($ok / ($elapsed / 3600))"); left=$(( size - ok - fail ))
    echo "rate $(printf '%.0f' "$rate") items/h, $left left, eta $(python3 -c "print(f'{$left / max($rate, 1e-9) * 60:.0f} min')")"
  fi
  if [ "$fail" -gt 0 ]; then
    echo "--- failed children (index, exit, reason) ---"
    aws batch list-jobs --region "$REGION" --array-job-id "$JOB" --job-status FAILED \
      --query 'jobSummaryList[].[arrayProperties.index, container.exitCode, statusReason]' --output text \
      | sort -n | head -40 | awk -F'\t' '{printf "  #%-5s exit %-4s %s\n", $1, $2, substr($3, 1, 90)}'
  fi
  echo "console: https://$REGION.console.aws.amazon.com/batch/home?region=$REGION#jobs/array/$JOB"
  [ "$status" = SUCCEEDED ] || [ "$status" = FAILED ]
}

if [ "$WATCH" = "--watch" ]; then
  until show; do echo; sleep 60; done
else
  show || true
fi
