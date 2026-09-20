#!/bin/bash
# Split a finished array job's failures into the two lists worth resubmitting.
#
#   scripts/batch/retry-list.sh <array-job-id> <items.txt> [out-prefix]
#
# Batch cannot change a job definition on retry, so an item killed for memory
# has to come back as a second submission against mapsnap-loc-fit-large. This
# works out which ones those were: loc-fit exits 1 either way, and only its
# log says whether a stage was killed (`exit -9`) or raised.
#
#   <prefix>-oom.txt     resubmit with JOBDEF=mapsnap-loc-fit-large
#   <prefix>-failed.txt  a real error; read one log before resubmitting
#
# Exit 3 (unprocessable) and 4 (inputs missing) are left out of both: neither
# is fixed by running it again.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
JOB=${1:?array job id}; LIST=${2:?the items.txt the job ran}; PREFIX=${3:-retry}
command -v jq > /dev/null || { echo "jq is needed" >&2; exit 1; }

: > "$PREFIX-oom.txt"; : > "$PREFIX-failed.txt"; skipped=0
while read -r index code; do
  item=$(sed -n "$((index + 1))p" "$LIST")
  case "$code" in
    3|4) skipped=$((skipped + 1)); continue ;;
  esac
  stream=$(aws batch describe-jobs --region "$REGION" --jobs "$JOB:$index" \
    --query 'jobs[0].container.logStreamName' --output text 2>/dev/null)
  tail=""
  if [ -n "$stream" ] && [ "$stream" != None ]; then
    tail=$(aws logs get-log-events --region "$REGION" --log-group-name /aws/batch/job \
      --log-stream-name "$stream" --limit 20 --query 'events[].message' --output text 2>/dev/null || true)
  fi
  # A stage killed by the kernel reports a negative code; 137 is the container
  # itself being OOM-killed, which Batch surfaces directly.
  if [ "$code" = 137 ] || grep -qE 'exit -9|Killed|MemoryError' <<< "$tail"; then
    echo "$item" >> "$PREFIX-oom.txt"
    printf '  %-22s index %-6s memory\n' "$item" "$index"
  else
    echo "$item" >> "$PREFIX-failed.txt"
    reason=$(grep -oE 'FAILED: [^|]{0,70}' <<< "$tail" | tail -1)
    printf '  %-22s index %-6s %s\n' "$item" "$index" "${reason:-exit $code}"
  fi
done < <(aws batch list-jobs --region "$REGION" --array-job-id "$JOB" --job-status FAILED \
           --query 'jobSummaryList[].[arrayProperties.index, container.exitCode]' --output text | sort -n)

oom=$(grep -c . "$PREFIX-oom.txt" || true); other=$(grep -c . "$PREFIX-failed.txt" || true)
echo
echo "$oom out of memory, $other other, $skipped not worth retrying (exit 3 or 4)"
[ "$oom" -gt 1 ] && echo "  PER_JOB=1 JOBDEF=mapsnap-loc-fit-large scripts/batch/submit.sh <run-tag> $PREFIX-oom.txt"
[ "$oom" = 1 ] && echo "  one item: mapsnap loc-fit --item $(cat "$PREFIX-oom.txt") --run-tag <run-tag> ... (an array needs two)"
exit 0
