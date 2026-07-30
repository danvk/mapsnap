"""Street-constraint georeferencing: solve, then adopt where the referee prefers it.

The production entry point for the street_solve channel (see street_solve.py for
the solver and street_solve_experiment.py for the underlying commands). Every
page with a key-map location prior gets a pose fitted from its street labels as
position+angle constraints against named OSM polylines — no intersections
required. On its own the channel is a coin flip against the incumbent, so a pose
is only adopted where an independent referee (osm_snap.evaluate_pose: road-
skeleton chamfer plus name alignment, evidence derived from neither channel)
prefers it by a clear margin. Measured over every disagreement in the twelve
truth volumes that picks the closer pose 85% of the time and never costs a page
its <=25 ft placement.

Writes pN.georef-streets.json sidecars for the adopted pages, deleting the
previous run's first. Build the volume IIIF with the streets-first hybrid glob
so the sidecars win where they exist:

    mapsnap iiif <ref> '<dir>/*.georef-streets.json,<dir>/*.georef-osm.json,<dir>/*.georef.json' ...

(`mapsnap fit` does this automatically.) The referee shares the snap channel's
machinery (P(road) maps, OSM rasters), so run this after `mapsnap snap` — the
incumbent it judges against is whatever the pipeline currently publishes,
snap's sidecars included.

Usage:
    mapsnap street-solve DIR
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit street-constraint poses for every key-map-prior page and adopt "
            "each one only where the independent referee prefers it. Writes "
            "pN.georef-streets.json sidecars; include them streets-first in the "
            "IIIF glob."
        )
    )
    parser.add_argument("dir", metavar="DIR", type=Path, help="Volume directory")
    args = parser.parse_args()

    from mapsnap.street_solve_experiment import ADOPT_GAP, cmd_candidates, cmd_select

    common = {"volume": str(args.dir), "gates": None}
    cmd_candidates(argparse.Namespace(**common, pages=None, truth_prior=False))
    cmd_select(argparse.Namespace(**common, adopt_gap=ADOPT_GAP))


if __name__ == "__main__":
    main()
