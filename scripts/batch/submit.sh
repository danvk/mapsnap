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
JOBDEF=${JOBDEF:-mapsnap-loc-fit}
TAG=${1:?run tag}; LIST=${2:?items.txt}
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)

SIZE=$(grep -c . "$LIST")
[ "$SIZE" -ge 2 ] || { echo "an array job needs at least 2 items ($SIZE in $LIST); use --item for one" >&2; exit 2; }
[ "$SIZE" -le 10000 ] || { echo "$SIZE items: Batch arrays cap at 10,000; split the list" >&2; exit 2; }
ITEMS=$BUCKET/_runs/$TAG/items.txt
aws s3 cp "$LIST" "$ITEMS" --only-show-errors --region "$REGION"

JOB_ID=$(aws batch submit-job --region "$REGION" \
  --job-name "fit-$TAG" --job-queue "$QUEUE" --job-definition "$JOBDEF" \
  --array-properties "size=$SIZE" \
  --parameters "items=$ITEMS,runTag=$TAG,bucket=$BUCKET" \
  --query jobId --output text)
echo "submitted fit-$TAG: $SIZE children, array job $JOB_ID"
echo "console: https://$REGION.console.aws.amazon.com/batch/home?region=$REGION#jobs/array/$JOB_ID"
echo "$JOB_ID" > "/tmp/mapsnap-batch-$TAG.jobid"
exec "$HERE/status.sh" "$JOB_ID"
