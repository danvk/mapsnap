#!/usr/bin/env python
"""Re-score a run's published p_good with the current confidence model, into a new IIIF file.

A run's ``mapsnap.iiif.json`` carries the p_good its reconcile computed with the
model of the day (mapsnap.confidence). When the model is retrained, the poses do
not change, only their scores -- and every input the model reads is already in
the run's ``pN.provenance.json`` and ``pN.panels.json``. So re-scoring a volume
needs no pipeline stage: read its provenance, score each placed page with the
new model, and replace the confidence entries of the matching annotations.

The input IIIF is never modified. The re-scored copy is written beside it as
``<name>.confidence-<model version>.iiif.json`` (``mapsnap.iiif.json`` ->
``mapsnap.confidence-corpus-v2.iiif.json``), for the publish step to rename.

    uv run python scripts/rescore_confidence.py VOLUME_DIR [...]
    uv run python scripts/rescore_confidence.py VOLUME_DIR --iiif corpus-v2.iiif.json
"""

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.confidence import MODEL_PATH, annotate
from mapsnap.make_iiif_georef import CONFIDENCE_LABELS, confidence_metadata


def output_path(iiif: Path, version: str) -> Path:
    """Where the re-scored copy of ``iiif`` goes: ``<stem>.confidence-<version>.iiif.json``."""
    stem = iiif.name.removesuffix(".iiif.json").removesuffix(".json")
    return iiif.with_name(f"{stem}.confidence-{version}.iiif.json")


def volume_confidences(volume: Path, model: dict) -> dict[str, dict]:
    """Page key (lowercased) -> the model's confidence record, for every placed page in a volume."""
    records = [
        json.loads(p.read_text()) for p in sorted(volume.glob("p*.provenance.json"))
    ]
    annotate(records, volume, model)
    return {
        record["stem"].lower(): record["confidence"]
        for record in records
        if record.get("confidence")
    }


def page_key(label: str) -> str | None:
    """The page key a mapsnap annotation's label ends with: '... | item p0028AL [2]' -> 'p0028AL__2'.

    Our labels end in the page's own stem, which can carry any LoC suffix
    (p0028AL, p0139snew), so this does not use utils.label_to_page_key, which
    parses OIM's labels and allows a single letter.
    """
    match = re.search(r"\s(p\S+?)(?:\s+\[(\d+)\])?$", label.strip())
    if match is None:
        return None
    return f"{match[1]}__{match[2]}" if match[2] else match[1]


def annotation_label(annotation: dict) -> str:
    """An annotation's label as text (IIIF allows a plain string or a language map)."""
    label = annotation.get("label") or ""
    if isinstance(label, dict):
        values = next(iter(label.values()), [""])
        return values[0] if values else ""
    return label


def rescore_page(page: dict, confidences: dict[str, dict]) -> tuple[dict, Counter[str]]:
    """A copy of an AnnotationPage with each annotation's confidence entries replaced.

    The new entries take the place of the old ``confidence`` entry, so the metadata
    keeps its order. An annotation whose page has no new score loses its old one
    rather than keep a score from another model. Returns the copy and counts of
    ``rescored`` / ``unscored`` annotations.
    """
    out = copy.deepcopy(page)
    counts: Counter[str] = Counter()
    for annotation in out.get("items") or []:
        metadata = annotation.get("metadata") or []
        key = page_key(annotation_label(annotation))
        confidence = confidences.get(key.lower()) if key else None
        new = confidence_metadata(confidence)
        at = next(
            (i for i, e in enumerate(metadata) if e.get("label") == "confidence"),
            len(metadata),
        )
        kept = [e for e in metadata[:at] if e.get("label") not in CONFIDENCE_LABELS]
        rest = [e for e in metadata[at:] if e.get("label") not in CONFIDENCE_LABELS]
        annotation["metadata"] = kept + new + rest
        counts["rescored" if new else "unscored"] += 1
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("volumes", type=Path, nargs="+")
    parser.add_argument(
        "--iiif",
        default="mapsnap.iiif.json",
        help="The run's annotation page in each volume.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    model = json.loads(args.model.read_text())
    totals: Counter[str] = Counter()
    for volume in args.volumes:
        iiif = volume / args.iiif
        if not iiif.exists():
            print(f"{volume.name}: no {args.iiif}, skipped", file=sys.stderr)
            totals["skipped"] += 1
            continue
        out = output_path(iiif, model["version"])
        page, counts = rescore_page(
            json.loads(iiif.read_text()), volume_confidences(volume, model)
        )
        out.write_text(json.dumps(page, indent=2))
        totals.update(counts)
        totals["volumes"] += 1
        print(
            f"{volume.name}: {counts['rescored']} rescored, {counts['unscored']} unscored -> {out.name}",
            file=sys.stderr,
        )
    print(
        f"{totals['volumes']} volumes ({totals['skipped']} skipped): "
        f"{totals['rescored']} annotations rescored, {totals['unscored']} unscored "
        f"with model {model['version']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
