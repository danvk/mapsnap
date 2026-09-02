"""Unified mapsnap command-line interface."""

import faulthandler
import importlib
import sys

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    # Pipelines
    "run-loc": (
        "mapsnap.pipeline_loc",
        "Full LOC pipeline: download, OCR, georeference, IIIF",
    ),
    "run-oim": (
        "mapsnap.pipeline_oim",
        "Full OIM pipeline: download, OCR, georeference, IIIF, compare",
    ),
    "fit": (
        "mapsnap.fit",
        "Georeference, build IIIF, and optionally compare (fast pipeline)",
    ),
    "rerun": (
        "mapsnap.pipeline_rerun",
        "Re-run split/keymap/ocr/adjacency/fit on downloaded volumes, reusing CRAFT boxes",
    ),
    "experiments": (
        "mapsnap.experiments",
        "Compare archived fit runs (experiments diff <id-a> <id-b>)",
    ),
    # Individual commands
    "craft": (
        "mapsnap.craft",
        "Run CRAFT text detection and cache <stem>.boxes.json (prerequisite for ocr/adjacency)",
    ),
    "ocr": ("mapsnap.detect_text", "Detect text regions in map images"),
    "georef": (
        "mapsnap.georef_from_labels",
        "Georeference a map from detected street labels",
    ),
    "keymap": (
        "mapsnap.keymap.pipeline",
        "Prepare key map(s): OCR+georef, page-number detection, and region segmentation",
    ),
    "keymap-detect": (
        "mapsnap.keymap.identify",
        "Identify which page(s) of a volume are key maps (from 25%-scale images)",
    ),
    "adjacency": (
        "mapsnap.page_adjacency",
        "Detect printed adjacent-sheet numbers and build a volume adjacency graph",
    ),
    "adjacency-gate": (
        "mapsnap.adjacency_gate",
        "Demote fits that contradict their printed mutual-adjacency claims",
    ),
    "snap": (
        "mapsnap.snap",
        "Geometry-first OSM snap: rescue unplaced pages, arbitrate and refine fits",
    ),
    "keymap-snap": (
        "mapsnap.keymap.snap",
        "Place pages by matching their P(road) map against the key map's",
    ),
    "street-solve": (
        "mapsnap.street_solve_run",
        "Street-constraint georeferencing: solve key-map-prior pages, adopt where the referee prefers",
    ),
    "iiif": (
        "mapsnap.make_iiif_georef",
        "Combine georeferences into a IIIF AnnotationPage",
    ),
    "archive": (
        "mapsnap.archive",
        "Archive an already-produced run's sidecars under artifacts/<tag>/",
    ),
    "compare": (
        "mapsnap.compare_iiif_georef",
        "Compare human vs computer IIIF georeferencing",
    ),
    "score": (
        "mapsnap.score",
        "Land-weighted success score vs truth (good share minus disaster share)",
    ),
    "download-osm": ("mapsnap.download_osm", "Download street data from OSM"),
    "osm-to-geojson": ("mapsnap.osm_to_centerlines", "Convert OSM data to GeoJSON."),
    "scale": ("mapsnap.scale_images", "Shrink images by a uniform amount."),
    "download-oim": (
        "mapsnap.download_oim_iiif",
        "Fetch all images for a Sanborn volume from OldInsuranceMaps.net",
    ),
    "download-loc": (
        "mapsnap.download_loc_iiif",
        "Fetch all images for a Sanborn volume from the Library of Congress (loc.gov)",
    ),
    "download-raw": (
        "mapsnap.download_raw",
        "Download the full-resolution LOC image for a scaled-down page JPG",
    ),
    "split-twopage": ("mapsnap.split_twopage", "Split two-page images in half"),
    "split": ("mapsnap.split", "Split map pages into individual panels"),
    "keymap-cartouche": (
        "mapsnap.keymap.cartouche",
        "Read cartouche words (GRAPHIC MAP OF VOLUMES, KEY, ...) on key-map sheets (#276)",
    ),
    "fix-truth-splits": (
        "mapsnap.fix_truth_splits",
        "Repair OIM-exported SvgSelectors for split pages (OIM#402)",
    ),
    "reconcile": (
        "mapsnap.reconcile",
        "Report-only joint arbitration of all candidate poses per volume (#270)",
    ),
    "oim-panels": (
        "mapsnap.oim_panels",
        "Build oim/pN.panels.json and cutlines.json from OIM's published region boundaries",
    ),
}

_cmd_width = max(len(cmd) for cmd in SUBCOMMANDS)
_commands_section = "\n".join(
    f"  {cmd:<{_cmd_width}}  {help_text}" for cmd, (_, help_text) in SUBCOMMANDS.items()
)
HELP = f"""Usage: mapsnap <command> [args...]

Commands:
{_commands_section}
"""


def main() -> None:
    # A native crash (SIGSEGV/SIGBUS/SIGFPE) in numpy, shapely, cv2 or torch kills
    # the process with no Python traceback at all -- #296 has taken out `snap`
    # three times in ~60 fits, and every report so far is just "rc=245".
    # faulthandler writes the C-level and Python stacks straight to fd 2 from the
    # signal handler, so the next occurrence names the line.
    faulthandler.enable()
    # Line-buffer stdout so progress SURVIVES such a crash. Piped output is
    # block-buffered by default, so the buffer dies with the process: the
    # crashed runs looked like `snap` had produced nothing before dying, when in
    # fact its output was simply lost. Without this the traceback above has no
    # context about which page was being processed.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(line_buffering=True)
        except ValueError:  # already-detached or non-text stream
            pass

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    if cmd not in SUBCOMMANDS:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print(f"Available commands: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        sys.exit(1)

    # Replace argv so the subcommand's argparse sees the right program name and args.
    sys.argv = [f"mapsnap {cmd}", *sys.argv[2:]]

    module_name, _ = SUBCOMMANDS[cmd]
    mod = importlib.import_module(module_name)
    mod.main()


if __name__ == "__main__":
    main()
