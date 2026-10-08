"""How likely is a placed page to be right? A calibrated confidence from the pipeline's own signals (#579).

``p_good`` is a logistic model's estimate that a pose lies within 25 ft of the
truth. Its inputs are only what reconcile records about the page in
``pN.provenance.json``:

- how constrained the fit is (effective and inlier GCPs, inlier streets);
- how well the pose agrees with OSM (P(road) verification, street-name and
  containment scores);
- whether independent channels landed in the same place (merged sources,
  rival poses within 50 / 200 m, the nearest rival);
- whether the printed neighbour stamps agree with the neighbours' placements;
- the key map's distance, and the scale's consistency with the volume's rungs;
- the chosen source and snap's verdict;
- for panels, the share of the sheet;
- two volume-level rates (share of pages placed, median verification).

Reconcile's own energies are left out on purpose: any change to reconcile
changes what they mean, and they add only 0.004 AUC.

Trained by ``scripts/train_confidence.py`` on corpus-v1 against OIM truth (400
volumes, 23,371 placed poses). Out-of-fold, with whole volumes held out, it
reaches AUC 0.88, against 0.87 for GCPs and verification together and 0.74
for GCPs alone. Poses scored 0.9-1.0 were 96% within 25 ft; those under 0.1
were 5%.

Panel scores are marked ``provisional``. Panels rank worse than whole pages
even within corpus-v1 (AUC 0.77 against 0.89), and #570 has since changed what
their signals mean: panels are now snapped and scored. Retrain on a
current-code corpus run before relying on them. Whole pages transfer: on the
20 standard volumes under current code (#570, #578), this model ranks pages at
AUC 0.91 (0.93 on the nine volumes it never saw), and is slightly cautious
(mean P(good) 85% against 88% observed).
"""

import json
import math
import statistics
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from shapely.geometry import Polygon

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "confidence.json"

NUMERIC = [
    "panel",
    "panel_fraction",
    "gcps",
    "fit_gcps",
    "inlier_intersections",
    "inlier_streets",
    "verification",
    "name",
    "containment",
    "keymap_dist_rel",
    "rung_distance",
    "off_rung",
    "note_mismatch",
    "ambiguous",
    "contradicted",
    "merged",
    "n_hypotheses",
    "agree_50m",
    "agree_200m",
    "nearest_other_m",
    "mutual_edges",
    "stamp_neighbours",
    "stamp_median_m",
    "stamp_agree_frac",
    "volume_placed_frac",
    "volume_median_verification",
]
CATEGORICAL = {
    "source": ["georef", "georef-snap", "georef-street", "snap", "streets"],
    "snap_verdict": [
        "none",
        "keep",
        "refine",
        "rescue",
        "abstain",
        "challenge",
        "rung-flip",
    ],
    "keymap": ["none", "georeferenced"],
    "fit_state": ["fitted", "nofit", "misscale", "none"],
}


def distance_m(a: list[float], b: list[float]) -> float:
    """Approximate ground distance between two (lon, lat) points, in metres."""
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * math.cos(lat), a[1] - b[1]) * 111_320


def source_kind(source: str) -> str:
    """The channel family of a hypothesis source: snap:2 -> snap, georef:contradicted -> georef."""
    if source.startswith("snap"):
        return "snap"
    if source.startswith("streets"):
        return "streets"
    return source.split(":")[0]


def panel_fraction(volume: Path, stem: str) -> float:
    """The share of its sheet a panel covers, from the sheet's panels.json; 1.0 for a whole page."""
    base, sep, index = stem.rpartition("__")
    if not sep:
        return 1.0
    path = volume / f"{base}.panels.json"
    if not path.exists():
        return 1.0
    doc = json.loads(path.read_text())
    rings = doc["panels"]
    k = int(index) - 1
    if not 0 <= k < len(rings):
        return 1.0
    return Polygon(rings[k]).area / (doc["width"] * doc["height"])


def volume_stats(records: list[dict]) -> dict[str, float | None]:
    """Volume-level rates over a volume's provenance records: share placed, median verification."""
    answerable = [r for r in records if r.get("decision") != "superseded"]
    placed = [r for r in records if r.get("decision") == "placed"]
    verifications = [
        h["verification"]
        for r in placed
        for h in r.get("hypotheses") or []
        if h.get("chosen") and h.get("verification") is not None
    ]
    return {
        "volume_placed_frac": len(placed) / max(1, len(answerable)),
        "volume_median_verification": statistics.median(verifications)
        if verifications
        else None,
    }


def page_features(
    record: dict, fraction: float, volume: Mapping[str, float | None]
) -> dict | None:
    """The model's inputs for one placed page, from its provenance record; None if nothing was chosen."""
    hypotheses = record.get("hypotheses") or []
    chosen = next((h for h in hypotheses if h.get("chosen")), None)
    if chosen is None or chosen.get("center") is None:
        return None
    evidence = record.get("evidence") or {}
    stamp = evidence.get("stamp_agreement") or {}
    rung = chosen.get("rung") or {}
    terms = chosen.get("terms") or {}
    others = [h for h in hypotheses if h is not chosen and h.get("center") is not None]
    distances = [distance_m(chosen["center"], h["center"]) for h in others]
    radius = evidence.get("keymap_radius_m")
    keymap_dist = chosen.get("keymap_dist_m")
    return {
        "source": source_kind(chosen["source"]),
        "snap_verdict": record.get("snap_verdict") or "none",
        "fit_state": evidence.get("fit_state") or "none",
        "keymap": evidence.get("keymap") or "none",
        "panel": "__" in record["stem"],
        "panel_fraction": fraction,
        "gcps": chosen.get("effective_gcps") or 0,
        "fit_gcps": max(
            [
                h.get("effective_gcps") or 0
                for h in hypotheses
                if h["source"].startswith("georef")
            ]
            or [0]
        ),
        "inlier_intersections": evidence.get("inlier_intersections") or 0,
        "inlier_streets": evidence.get("inlier_streets") or 0,
        "verification": chosen.get("verification"),
        "name": chosen.get("name"),
        "containment": chosen.get("containment"),
        "keymap_dist_rel": keymap_dist / radius
        if keymap_dist is not None and radius
        else None,
        "rung_distance": rung.get("rung_distance"),
        "off_rung": rung.get("verdict") == "between rungs",
        "note_mismatch": rung.get("verdict") == "contradicts printed note",
        "ambiguous": "ambiguity" in terms,
        "contradicted": "contradicted" in terms,
        "merged": len(record.get("merged") or []),
        "n_hypotheses": len(others),
        "agree_50m": sum(d <= 50 for d in distances),
        "agree_200m": sum(d <= 200 for d in distances),
        "nearest_other_m": min(distances) if distances else None,
        "mutual_edges": evidence.get("mutual_edges") or 0,
        "stamp_neighbours": stamp.get("neighbours") or 0,
        "stamp_median_m": stamp.get("median_m"),
        "stamp_agree_frac": stamp["agree_100m"] / stamp["neighbours"]
        if stamp.get("neighbours")
        else None,
        **volume,
    }


def feature_vector(features: dict) -> tuple[list[float | None], list[str]]:
    """(values, column names): numeric features as floats (None if missing), categoricals one-hot."""
    values: list[float | None] = [
        None if features.get(c) is None else float(features[c]) for c in NUMERIC
    ]
    names = list(NUMERIC)
    for column, levels in CATEGORICAL.items():
        values += [float(features.get(column) == level) for level in levels]
        names += [f"{column}={level}" for level in levels]
    return values, names


def p_good(features: dict, model: dict) -> float:
    """The model's probability that this pose is within 25 ft of the truth.

    ``model`` is a ``models/confidence.json`` document: per column a fill value
    for a missing input, whether its missingness is itself a feature, a mean and
    a scale to standardize by, and a coefficient.
    """
    values, names = feature_vector(features)
    z = model["intercept"]
    for name, value in zip(names, values, strict=True):
        column = model["columns"][name]
        x = column["fill"] if value is None else value
        z += column["coef"] * (x - column["mean"]) / column["scale"]
        if "missing_coef" in column:
            z += column["missing_coef"] * (1.0 if value is None else 0.0)
    return 1.0 / (1.0 + math.exp(-z))


@cache
def load_model(path: Path = MODEL_PATH) -> dict | None:
    """The shipped confidence model, or None if it is absent."""
    return json.loads(path.read_text()) if path.exists() else None


def annotate(records: list[dict], volume: Path, model: dict | None = None) -> None:
    """Add ``confidence`` to every placed page's provenance record, in place.

    ``{"p_good": 0.93, "provisional": false, "model": "<version>"}``; panels are
    provisional until the model is retrained on current-code panels.
    """
    model = model if model is not None else load_model()
    if model is None:
        return
    stats = volume_stats(records)
    for record in records:
        if record.get("decision") != "placed":
            continue
        features = page_features(record, panel_fraction(volume, record["stem"]), stats)
        if features is None:
            continue
        record["confidence"] = {
            "p_good": round(p_good(features, model), 3),
            "provisional": bool(features["panel"]),
            "model": model["version"],
        }
