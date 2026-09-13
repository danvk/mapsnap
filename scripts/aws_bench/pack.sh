#!/bin/bash
# Build and upload the benchmark payload (#354): one volume's 25%-scale parent pages,
# its centerlines and volume metadata, one raw key-map sheet, and the EasyOCR weights.
#
#   scripts/aws_bench/pack.sh                                  # Hudson, ~190 MB
#   scripts/aws_bench/pack.sh data/detroit_mi_1921_vol_1       # another volume
#
# Split panels and every sidecar are left out: the instance runs craft/ocr from
# scratch, which is the point. Uploads with the mapsnap profile (it already writes
# the bucket).
set -euo pipefail

VOLUME=${1:-data/hudson_co_nj_1950_vol_9}
BUCKET=${BUCKET:-mapsnap-sanborn}
PREFIX=${PREFIX:-_bench}
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
NAME=$(basename "$VOLUME")
OUT=$(mktemp -d)/hudson-bench.tar.gz

RAW=$(ls "$VOLUME"/raw/p0*.jpg "$VOLUME"/raw/p[A-Za-z].jpg 2>/dev/null | head -1)
if [ -z "$RAW" ]; then
  echo "no raw key-map sheet under $VOLUME/raw/" >&2
  exit 1
fi
PAGES=$(cd "$VOLUME" && ls p*.jpg | grep -v __)
# shellcheck disable=SC2086  # PAGES is a whitespace-separated list by design
tar czf "$OUT" \
  -C "$(dirname "$VOLUME")" $(for p in $PAGES; do echo "$NAME/$p"; done) \
     "$NAME/centerlines.geojson" "$NAME/mapsnap.json" "$NAME/raw/$(basename "$RAW")" \
  -C "$HOME/.EasyOCR" model
ls -lh "$OUT"
aws s3 cp "$OUT" "s3://$BUCKET/$PREFIX/hudson-bench.tar.gz"
rm -rf "$(dirname "$OUT")"
