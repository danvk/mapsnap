"""Unified mapsnap command-line interface."""

import importlib
import sys

subcommands = {
    "ocr": "mapsnap.detect_text",
    "georef": "mapsnap.georef_from_labels",
    "iiif": "mapsnap.make_iiif_georef",
    "compare": "mapsnap.compare_iiif_georef",
}

HELP = """Usage: mapsnap <command> [args...]

Commands:
  ocr      Detect text regions in map images
  georef   Georeference a map from detected street labels
  iiif     Combine georeferences into a IIIF AnnotationPage
  compare  Compare human vs computer IIIF georeferencing
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    if cmd not in subcommands:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print(f"Available commands: {', '.join(subcommands)}", file=sys.stderr)
        sys.exit(1)

    # Replace argv so the subcommand's argparse sees the right program name and args.
    sys.argv = [f"mapsnap {cmd}", *sys.argv[2:]]

    mod = importlib.import_module(subcommands[cmd])
    mod.main()


if __name__ == "__main__":
    main()
