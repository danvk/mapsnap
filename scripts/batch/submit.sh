#!/bin/bash
# Submit one corpus fit as an AWS Batch array job: one child per item.
#
#   scripts/batch/submit.sh <run-tag> <items.txt>
#   scripts/batch/submit.sh batch-test-200 /tmp/test200.txt
#
# <items.txt> is one item id per line (sanborn02404_004). It is uploaded to
# s3://mapsnap-sanborn/_runs/<run-tag>/items.txt and child N runs line N:
# loc-fit reads AWS_BATCH_JOB_ARRAY_INDEX (see `mapsnap loc-fit --items`).
# Outputs land under <item>/runs/<run-tag>/ exactly as a fleet run's do, and
# an item already finished under that tag is skipped (exit 0), so resubmitting
# the same list after a failure re-runs only what is left.
#
# Prints the array job id and hands off to status.sh.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
BUCKET=${BUCKET:-s3://mapsnap-sanborn}
QUEUE=${QUEUE:-mapsnap-fit}
# The ~1.5% of items whose OCR vocabulary needs more than 8 GB go to the
# 16 GB definition; a run only learns which those are by killing them, so
# this is how the retry run is submitted, not something to guess up front.
JOBDEF=${JOBDEF:-mapsnap-loc-fit}
TAG=${1:?run tag}; LIST=${2:?items.txt}
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)

# Items per child. A Batch array caps at 10,000 children and the mirror holds
# 35,159 items, so the corpus does not fit one item to a child at all; 8 also
# gives loc-fit's prefetch something to overlap, which a one-item process has
# not had since the queue went away.
PER_JOB=${PER_JOB:-1}
COUNT=$(grep -c . "$LIST")
SIZE=$(( (COUNT + PER_JOB - 1) / PER_JOB ))
[ "$SIZE" -ge 2 ] || { echo "an array job needs at least 2 children ($COUNT items at $PER_JOB per job); use --item for one" >&2; exit 2; }
[ "$SIZE" -le 10000 ] || { echo "$COUNT items at $PER_JOB per job needs $SIZE children; Batch arrays cap at 10,000 -- raise PER_JOB" >&2; exit 2; }
# Key the list by its own content, not by the run tag. A corpus run is filled
# in several passes under one tag -- a sample, then the rest, then a mop-up --
# and a fixed key meant the second pass overwrote the first's record. That is
# not only lost bookkeeping: a child of the earlier array retried after the
# overwrite would read the new list and fit the wrong items.
LIST_ID=$(cksum < "$LIST" | cut -d' ' -f1)
ITEMS=$BUCKET/_runs/$TAG/items-$LIST_ID.txt
aws s3 cp "$LIST" "$ITEMS" --only-show-errors --region "$REGION"

JOB_ID=$(aws batch submit-job --region "$REGION" \
  --job-name "fit-$TAG" --job-queue "$QUEUE" --job-definition "$JOBDEF" \
  --array-properties "size=$SIZE" \
  --parameters "items=$ITEMS,runTag=$TAG,bucket=$BUCKET,itemsPerJob=$PER_JOB" \
  --query jobId --output text)
echo "submitted fit-$TAG: $SIZE children, array job $JOB_ID"
echo "console: https://$REGION.console.aws.amazon.com/batch/home?region=$REGION#jobs/array/$JOB_ID"
echo "$JOB_ID" > "/tmp/mapsnap-batch-$TAG.jobid"
exec "$HERE/status.sh" "$JOB_ID"
