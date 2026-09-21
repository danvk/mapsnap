#!/bin/bash
# Split a finished array job's failures into the two lists worth resubmitting.
#
#   scripts/batch/retry-list.sh <array-job-id> [out-prefix]
#
# Batch cannot change a job definition on retry, so an item killed for memory
# has to come back as a second submission against mapsnap-loc-fit-large. This
# works out which ones those were.
#
#   <prefix>-oom.txt     resubmit with JOBDEF=mapsnap-loc-fit-large
#   <prefix>-failed.txt  a real error; read one log before resubmitting
#
# The failing items come out of the logs rather than off the items list, which
# is what makes this right for a chunked run: a child holding eight items
# reports the one that died, and its index says nothing about which. The other
# seven are already published and are skipped on the retry.
#
# Exit 3 (unprocessable) and 4 (inputs missing) are left out of both: neither
# is fixed by running it again.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
JOB=${1:?array job id}; PREFIX=${2:-retry}

: > "$PREFIX-oom.txt"; : > "$PREFIX-failed.txt"; skipped=0; unnamed=0

while read -r index code; do
  case "$code" in
    3|4) skipped=$((skipped + 1)); continue ;;
  esac
  stream=$(aws batch describe-jobs --region "$REGION" --jobs "$JOB:$index" \
    --query 'jobs[0].container.logStreamName' --output text 2>/dev/null)
  tail=""
  if [ -n "$stream" ] && [ "$stream" != None ]; then
    tail=$(aws logs get-log-events --region "$REGION" --log-group-name /aws/batch/job \
      --log-stream-name "$stream" --limit 60 --query 'events[].message' --output text 2>/dev/null || true)
  fi

  # Every item that died names itself: "<item>: FAILED: mapsnap <stage> failed
  # (exit N): ...". A chunk can lose more than one.
  named=0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    named=1
    item=${line%%:*}
    # An allocation the machine could never satisfy is a bug, not a ceiling: a
    # bad scale prior asked osm_rasters for a 120 GiB square on the 2026-09-20
    # sample, and no job definition has that. Those belong with the errors,
    # where someone will read one, rather than in a retry that fails again.
    if grep -qE 'Unable to allocate [0-9.]+ [GT]iB|_ArrayMemoryError' <<< "$line"; then
      echo "$item" >> "$PREFIX-failed.txt"
      printf '  %-22s child %-5s impossible allocation, not a ceiling: %s\n' "$item" "$index" \
        "$(grep -oE 'Unable to allocate [0-9.]+ [KMGT]iB[^|]*' <<< "$line" | head -1)"
      continue
    fi
    # Memory shows up three ways: a stage killed outright (-9), a stage whose
    # own child was killed and whose status came back through the shell's
    # 256-N convention (247), and the container itself being OOM-killed (137).
    if grep -qE 'exit (-9|247|137)\)|Killed|MemoryError|Cannot allocate memory' <<< "$line"; then
      echo "$item" >> "$PREFIX-oom.txt"
      printf '  %-22s child %-5s memory   %s\n' "$item" "$index" "$(grep -oE 'mapsnap [a-z-]+ failed \(exit -?[0-9]+\)' <<< "$line" | head -1)"
    else
      echo "$item" >> "$PREFIX-failed.txt"
      printf '  %-22s child %-5s %s\n' "$item" "$index" "$(grep -oE 'FAILED: .{0,64}' <<< "$line" | head -1)"
    fi
  done < <(tr '\t' '\n' <<< "$tail" | grep -E '^[A-Za-z0-9_]+: FAILED:' || true)

  if [ "$named" = 0 ]; then
    # No item named itself: the container died before it could say so (a spot
    # reclamation that outlived its retries, or a log that has aged out).
    unnamed=$((unnamed + 1))
    printf '  %-22s child %-5s exit %s, nothing named in the log\n' "(unknown)" "$index" "$code"
  fi
done < <(aws batch list-jobs --region "$REGION" --array-job-id "$JOB" --job-status FAILED \
           --query 'jobSummaryList[].[arrayProperties.index, container.exitCode]' --output text | sort -n)

sort -u -o "$PREFIX-oom.txt" "$PREFIX-oom.txt"
sort -u -o "$PREFIX-failed.txt" "$PREFIX-failed.txt"
oom=$(grep -c . "$PREFIX-oom.txt" || true); other=$(grep -c . "$PREFIX-failed.txt" || true)
echo
echo "$oom out of memory, $other other, $skipped not worth retrying (exit 3 or 4), $unnamed unattributed"
if [ "$oom" -gt 1 ]; then
  echo "  PER_JOB=1 JOBDEF=mapsnap-loc-fit-large scripts/batch/submit.sh <run-tag> $PREFIX-oom.txt"
elif [ "$oom" = 1 ]; then
  echo "  one item, and an array needs two:"
  echo "  mapsnap loc-fit --item $(cat "$PREFIX-oom.txt") --run-tag <run-tag> --counties ... (or add it to the next run)"
fi
exit 0
