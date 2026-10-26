#!/bin/bash
# Seed a new run tag from an earlier one, so loc-fit re-runs only the last stages.
#
#   scripts/batch/seed-run.sh <dest-tag> <items.txt> [<default-source-tag>]
#   scripts/batch/seed-run.sh corpus-v2.2 rerun.txt corpus-v2
#
# Each line of <items.txt> is an item id, optionally followed by a tab and the
# run tag to copy from (items published from different runs need different
# sources); the default source tag covers the rest.
#
# For each item, a server-side copy of runs/<source>/ into runs/<dest>/, without
# the outputs the rerun must regenerate: the done marker (mapsnap.iiif.json,
# without which loc-fit would call the item finished), the key-map IIIF, the old
# manifest and fit.log. What comes along -- sidecars, snap and street-solve
# candidates, checkpoints/ and .pipeline/ markers -- makes loc-fit skip
# keymap-detect, adjacency, keymap and ocr, and `fit --resume` skip georef, snap
# and street-solve, so a job submitted under <dest-tag> runs split, craft (both
# fast), region, reconcile and iiif. Nothing is rewritten in place.
#
# Never seed into an item that has already run under <dest-tag>: sync re-copies
# every file whose size differs, so it would put the source run's sidecars back
# over the new run's (it did, to NY 1923's corpus-v2.2 pilot). Items with a done
# marker under <dest-tag> are refused, and nothing should be submitted under
# <dest-tag> until this has finished.
#
# --copy-props none: a copy otherwise reads each object's tags, which the
# mapsnap-mirror user may not (GetObjectTagging is AccessDenied).
# Eight items at a time; the copies run inside S3. Idempotent.
set -euo pipefail
DEST=${1:?destination run tag}; LIST=${2:?items list}; DEFAULT=${3:-corpus-v2}
export PATH=/opt/homebrew/bin:$PATH AWS_PROFILE=${AWS_PROFILE:-mapsnap}
BUCKET=${BUCKET:-s3://mapsnap-sanborn}
MAPPING=${MAPPING:-$HOME/Downloads/loc-sanborn-maps.mapping.tsv}
[ -s "$MAPPING" ] || { echo "no mapping TSV at $MAPPING (it's also at $BUCKET/loc-sanborn-maps.mapping.tsv)" >&2; exit 1; }

seed_one() {
  item=$(printf '%s' "$1" | cut -f1)
  source=$(printf '%s' "$1" | cut -s -f2)
  source=${source:-$DEFAULT}
  prefix=$(awk -F'\t' -v i="$item" -v b="$BUCKET" '$1==i {print b"/by-state/"$2"/"$3"/"$1; exit}' "$MAPPING")
  [ -n "$prefix" ] || { echo "$item: not in the mapping" >&2; return 1; }
  if aws s3 ls "$prefix/runs/$DEST/mapsnap.iiif.json" > /dev/null 2>&1; then
    echo "$(date +%T) $item REFUSED: already run under $DEST" >&2
    return 0
  fi
  aws s3 sync "$prefix/runs/$source/" "$prefix/runs/$DEST/" --only-show-errors --region us-west-2 --copy-props none \
    --exclude "mapsnap.iiif.json" --exclude "mapsnap.keymap.iiif.json" \
    --exclude "artifacts/mapsnap/*" --exclude "fit.log" \
    && echo "$(date +%T) $item <- $source" || echo "$(date +%T) $item FAILED"
}
export -f seed_one
export DEST DEFAULT BUCKET MAPPING
grep . "$LIST" | tr '\n' '\0' | xargs -0 -P 8 -n 1 bash -c 'seed_one "$0"'
echo "SEEDED"
