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
shopt -s nullglob

VOLUME=${1:-data/hudson_co_nj_1950_vol_9}
BUCKET=${BUCKET:-mapsnap-sanborn}
PREFIX=${PREFIX:-_bench}
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
NAME=$(basename "$VOLUME")
OUT=$(mktemp -d)/hudson-bench.tar.gz

# The raw key-map sheet: p0-ish stems first, then a lettered one (LA's raw/pa.jpg).
SHEETS=("$VOLUME"/raw/p0*.jpg "$VOLUME"/raw/p[A-Za-z].jpg)
if [ ${#SHEETS[@]} -eq 0 ]; then
  echo "no raw key-map sheet under $VOLUME/raw/" >&2
  exit 1
fi
RAW=${SHEETS[0]}

# Parent pages only (no split panels), plus the volume's inputs, as tar members
# relative to the data directory so the archive unpacks as <volume name>/...
FILES=()
for page in "$VOLUME"/p*.jpg; do
  case "$(basename "$page")" in
    *__*) ;;
    *) FILES+=("$NAME/$(basename "$page")") ;;
  esac
done
if [ ${#FILES[@]} -eq 0 ]; then
  echo "no p*.jpg pages under $VOLUME" >&2
  exit 1
fi
FILES+=("$NAME/centerlines.geojson" "$NAME/mapsnap.json" "$NAME/raw/$(basename "$RAW")")

tar czf "$OUT" -C "$(dirname "$VOLUME")" "${FILES[@]}" -C "$HOME/.EasyOCR" model
echo "$((${#FILES[@]} - 3)) pages + $(basename "$RAW") + EasyOCR weights: $(du -h "$OUT" | cut -f1)"
aws s3 cp "$OUT" "s3://$BUCKET/$PREFIX/hudson-bench.tar.gz"
rm -rf "$(dirname "$OUT")"
