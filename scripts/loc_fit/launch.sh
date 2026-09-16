#!/bin/bash
# Launch one loc-fit run: its own queue, its own S3 directory, one tag (#354).
#
#   scripts/loc_fit/launch.sh --run-tag v1.3 --fill --limit 200 --shards 2
#   scripts/loc_fit/launch.sh --run-tag v1.3 --fill --shards 8
#   scripts/loc_fit/launch.sh --run-tag v1.3 --shards 4        # more workers, queue already filled
#   scripts/loc_fit/launch.sh --run-tag v1.4 --ocr-from v1.3 --fill --shards 8
#
# A run is (queue, tag, S3 directory) and this script is the only place the tag
# is typed. Outputs land in <item>/runs/<tag>/, so a pilot cannot collide with
# the full pass and two runs at the same commit can be compared.
#
# One queue PER RUN, which is why the queue is named after the tag: SQS has no
# selective receive, so a worker cannot decline a message meant for another run.
# Merely passing one over counts against maxReceiveCount, and three of those
# dead-letter a message nothing ever worked on.
#
# The EC2 machinery is shared with loc-craft, so this delegates rather than
# duplicating it; only the job, the queue and the tag differ.
set -euo pipefail

RUN_TAG=""
OCR_FROM=""
SHARDS=1
WORKERS=1
LIMIT=""
MANIFEST=""
# An item waiting on loc-craft is RELEASED, not retired, and a release counts as
# a receive: at the default 3 a not-ready item dead-letters after three passes.
# Fine once craft is done, too fast while it is still running, hence the flag.
MAX_RECEIVES=3
FILL=0
BUCKET=${BUCKET:-s3://mapsnap-sanborn}
# `loc-fit` REQUIRES --counties: without a county extract an item cannot read a
# street name, so there is no useful default inside the command and it refuses
# to start. The bucket's staged copies are that default here.
COUNTIES=""
DRY_RUN=0
# The lease the worker renews while an item runs (work_queue.lease). Shorter
# than the longest item on purpose: the heartbeat covers the long ones, and a
# short lease means a dead worker's item comes back sooner.
VISIBILITY=1800
# The queue must be created in the region the fleet will run in, so --region is
# parsed here rather than passed blindly through.
REGION=${AWS_REGION:-us-west-2}
PASS_THROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --ocr-from) OCR_FROM="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --max-receives) MAX_RECEIVES="$2"; shift 2 ;;
    --counties) COUNTIES="$2"; shift 2 ;;   # space-separated paths or s3:// URLs
    --visibility) VISIBILITY="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --fill) FILL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    # Handed to the shared launcher untouched. A whitelist, NOT a catch-all:
    # a blind `--*) PASS_THROUGH+=("$1" "$2")` swallows the next argument, and
    # a typo before --dry-run therefore ate it and ran the thing for real,
    # creating a queue. Every flag here takes a value.
    --only|--on-demand-from|--instance-type|--git-ref)
      if [ $# -lt 2 ]; then echo "$1 needs a value" >&2; exit 2; fi
      PASS_THROUGH+=("$1" "$2"); shift 2 ;;
    --job|--extra-args)
      echo "$1 is set by this script: use --run-tag and --ocr-from" >&2; exit 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_TAG" ]; then
  echo "--run-tag is required: it names the queue, the S3 directory and the" >&2
  echo "run recorded in every manifest and annotation page. Cut the release" >&2
  echo "first and use its tag." >&2
  exit 2
fi
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*)
    echo "run tag '$RUN_TAG' must be letters, digits, dot, dash or underscore:" >&2
    echo "it becomes an S3 path segment." >&2
    exit 2 ;;
esac

# SQS queue names allow only alphanumerics, dash and underscore -- no dots -- so
# a perfectly good tag like "v1.3" cannot be one verbatim. The S3 directory and
# the recorded provenance keep the tag as typed; only the queue name is folded.
QUEUE_NAME="mapsnap-fit-$(printf '%s' "$RUN_TAG" | tr -c 'A-Za-z0-9_-' '-')"

export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
export AWS_REGION=$REGION

HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)
REPO=$(cd -- "$HERE/../.." > /dev/null && pwd -P)

run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '  would run:'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$@"
}

echo "run tag   $RUN_TAG"
echo "region    $REGION"
echo "queue     $QUEUE_NAME"
[ -n "$OCR_FROM" ] && echo "ocr from  $OCR_FROM (reads reused where the county extract matches)"

# Idempotent: SQS hands back the existing queue when the attributes match, so
# adding workers to a running pass is the same command without --fill.
if [ "$DRY_RUN" = 1 ]; then
  run uv run --directory "$REPO" mapsnap work-queue create \
    --name "$QUEUE_NAME" --visibility "$VISIBILITY" --max-receives "$MAX_RECEIVES"
  QUEUE_URL="https://sqs.example/DRY-RUN/$QUEUE_NAME"
else
  QUEUE_URL=$(uv run --directory "$REPO" mapsnap work-queue create \
    --name "$QUEUE_NAME" --visibility "$VISIBILITY" \
    --max-receives "$MAX_RECEIVES" \
    | awk '$1 == "queue" {print $2}')
  if [ -z "$QUEUE_URL" ]; then
    echo "could not create or find queue $QUEUE_NAME" >&2
    exit 1
  fi
fi
echo "queue url $QUEUE_URL"

# Every message carries the tag, so a worker takes the run from the work rather
# than from its own flags and the two cannot disagree.
if [ "$FILL" = 1 ]; then
  fill_args=(--url "$QUEUE_URL" --run-tag "$RUN_TAG")
  [ -n "$LIMIT" ] && fill_args+=(--limit "$LIMIT")
  # A sample run fills from its own manifest; the workers keep the full one,
  # since the queue names the items and the manifest only resolves them.
  [ -n "$MANIFEST" ] && fill_args+=(--manifest "$MANIFEST")
  run uv run --directory "$REPO" mapsnap work-queue fill "${fill_args[@]}"
fi

if [ -z "$COUNTIES" ]; then
  COUNTIES="$BUCKET/_craft/items.tsv $BUCKET/_craft/city-items.tsv"
fi
WORKER_ARGS="--queue $QUEUE_URL --run-tag $RUN_TAG"
[ -n "$OCR_FROM" ] && WORKER_ARGS="$WORKER_ARGS --ocr-from $OCR_FROM"
# Last, because --counties takes one or more values and would otherwise swallow
# the flag that followed it.
WORKER_ARGS="$WORKER_ARGS --counties $COUNTIES"

# The worker's arguments are only checked when a worker runs them, which is on
# an instance, after boot -- so a missing required flag costs three instances
# and a silent queue. Parse them here instead, where it costs nothing.
if ! uv run --directory "$REPO" mapsnap loc-fit --help > /dev/null 2>&1; then
  echo "cannot run 'mapsnap loc-fit' locally to validate worker arguments" >&2
  exit 1
fi
# shellcheck disable=SC2086  # WORKER_ARGS is a flag string by design
if ! validation=$(uv run --directory "$REPO" mapsnap loc-fit --check-args $WORKER_ARGS 2>&1); then
  echo "the worker command these flags build is not valid:" >&2
  echo "$validation" >&2
  exit 2
fi

run "$HERE/../loc_craft/launch.sh" \
  --job loc-fit \
  --shards "$SHARDS" \
  --workers "$WORKERS" \
  --region "$REGION" \
  --extra-args "$WORKER_ARGS" \
  ${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}

echo
echo "watch it drain:  uv run mapsnap work-queue status --url $QUEUE_URL"
echo "outputs:         s3://mapsnap-sanborn/by-state/*/*/*/runs/$RUN_TAG/"
