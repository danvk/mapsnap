"""Geometry-first OSM snap: rescue, arbitrate, and refine a volume's fits.

The production entry point for the osm_snap channel (see osm_snap.py for the
matcher and osm_snap_experiment.py for the underlying commands). Matches each
page's road-UNet P(road) map against OSM centerlines rasterized in a local
metre frame — no street-name OCR required — and writes pN.georef-osm.json
sidecars for three kinds of placements:

  - RESCUE: pages the RANSAC georeferencer left unplaced (nofit/misscale/
    1gcp/outlier variants, plus split panels), placed from their key-map
    location via rotation/scale prior ladders and gated selection;
  - ARBITRATION: replacements for placed fits that OSM actively contradicts
    (incumbent verification < 0.1) when a confident challenger disagrees by
    >100 ft and wins the shared-evidence head-to-head;
  - REFINEMENT: adoption of a challenger that AGREES with a placed fit
    (<100 ft) and beats its verification by a clear margin — the mid-tier
    precision polish (agreeing snap poses beat 25-100 ft RANSAC fits 85-93%
    of the time).

Everything here is truth-free; main.iiif.json, when present, only annotates
diagnostics. Road-UNet P(road) maps are inferred on demand (cached under
artifacts/edge_join/roadprob/). Build the volume IIIF with the osm-first
hybrid glob so the sidecars win where they exist:

    mapsnap iiif <ref> '<dir>/*.georef-osm.json,<dir>/*.georef.json' ...

(`mapsnap fit` does this automatically.) First run costs UNet inference plus
NCC matching (roughly 10-30 minutes per volume); candidates are cached in
artifacts/osm_snap/candidates.jsonl and reruns are seconds.

Usage:
    mapsnap snap DIR [--rescue-only] [--recompute]
"""

import argparse
from pathlib import Path

# The production gates, frozen from the dev-volume sweeps (see the osm-snap
# PR for the calibration story): argmax rescue gate + distinct margin, the
# volume-energy conservative elbow (VOLUME_MODE_GATE), the arbitration score
# gate, and the refinement verification margin (REFINE_VER_MARGIN).
GATE_SCORE = 1.25
GATE_MARGIN = 0.25
ARBITRATE_GATE = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Geometry-first OSM snap: rescue unplaced pages, arbitrate "
            "OSM-contradicted fits, and refine mid-tier fits. Writes "
            "pN.georef-osm.json sidecars; include them osm-first in the "
            "IIIF glob."
        )
    )
    parser.add_argument("dir", metavar="DIR", type=Path, help="Volume directory")
    parser.add_argument(
        "--rescue-only",
        action="store_true",
        help=(
            "Only place unplaced pages; skip the fitted-page candidate pass "
            "(and with it arbitration and refinement). Much cheaper."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached candidates and re-match every target page.",
    )
    args = parser.parse_args()

    from mapsnap.osm_snap_experiment import (
        cmd_candidates,
        cmd_materialize,
        cmd_select,
    )

    mode = "union" if args.rescue_only else "arbitrate"
    cmd_candidates(
        args.dir,
        pages=None,
        all_pages=not args.rescue_only,
        limit=None,
        recompute=args.recompute,
        vis=False,
    )
    cmd_select(args.dir, mode, GATE_SCORE, GATE_MARGIN, ARBITRATE_GATE)
    cmd_materialize(args.dir, mode)


if __name__ == "__main__":
    main()
