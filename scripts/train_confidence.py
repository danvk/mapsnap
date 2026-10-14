#!/usr/bin/env python
"""Train the per-pose confidence model (mapsnap.confidence, #579) from a run plus OIM truth.

Each VOLUME_DIR holds one volume's run outputs beside its truth:
``pN.provenance.json`` and ``pN.panels.json`` from the run, the run's
published annotation page (``--iiif``, default ``mapsnap.iiif.json``),
``main.iiif.json`` (OIM truth) and, for split truth, ``oim/``. A placed page is
labelled by its RMSE against the truth pages ``mapsnap compare`` pairs it with
(the worst one, when it is paired with several).

Prints grouped cross-validation (whole volumes held out) and writes the model as
JSON. ``models/confidence.json`` was trained on corpus-v1 in the 400 OIM-truth
volumes:

    uv run python scripts/train_confidence.py ~/Documents/mapsnap/quality/train/* \\
        --iiif corpus-v1.iiif.json --version corpus-v1 --out models/confidence.json
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.compare_iiif_georef import compare_pages
from mapsnap.confidence import (
    feature_vector,
    page_features,
    panel_fraction,
    volume_stats,
)

GOOD_FT = 25.0  # what "good" means: the metric's threshold


def volume_rows(volume: Path, iiif_name: str) -> list[tuple[dict, float]]:
    """(features, worst RMSE in ft) for every placed page of one volume that truth grades."""
    rows, _ = compare_pages(
        volume / "main.iiif.json", volume / iiif_name, oim_dir=volume / "oim"
    )
    worst: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.get("rmse_ft") is not None and row.get("gen_page_key"):
            worst[row["gen_page_key"]] = max(worst[row["gen_page_key"]], row["rmse_ft"])
    records = [json.loads(p.read_text()) for p in volume.glob("p*.provenance.json")]
    stats = volume_stats(records)
    out = []
    for record in records:
        if record.get("decision") != "placed" or record["stem"] not in worst:
            continue
        features = page_features(record, panel_fraction(volume, record["stem"]), stats)
        if features is not None:
            out.append((features, worst[record["stem"]]))
    return out


def design(raw: np.ndarray, columns: dict | None = None) -> tuple[np.ndarray, dict]:
    """Standardized design matrix plus missingness indicators, and the column transforms.

    ``raw`` has NaN for missing inputs. Fitting (``columns`` None) computes each
    column's fill (median), mean and scale, and gives a column whose input was
    ever missing an indicator; applying reuses them.
    """
    if columns is None:
        columns = {}
        for j in range(raw.shape[1]):
            values = raw[:, j]
            present = values[~np.isnan(values)]
            fill = float(np.median(present)) if len(present) else 0.0
            filled = np.where(np.isnan(values), fill, values)
            scale = float(filled.std()) or 1.0
            columns[j] = {
                "fill": fill,
                "mean": float(filled.mean()),
                "scale": scale,
                "indicator": bool(np.isnan(values).any()),
            }
    parts = []
    for j, column in columns.items():
        values = raw[:, j]
        filled = np.where(np.isnan(values), column["fill"], values)
        parts.append((filled - column["mean"]) / column["scale"])
    for j, column in columns.items():
        if column["indicator"]:
            parts.append(np.isnan(raw[:, j]).astype(float))
    return np.column_stack(parts), columns


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 2.0, iterations: int = 30
) -> np.ndarray:
    """L2-penalized logistic regression by Newton's method; returns [intercept, *coefficients].

    The intercept is not penalized. ``l2`` is the penalty on the squared norm of
    the coefficients (sklearn's 1/C).
    """
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    penalty = np.full(A.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(A @ w)))
        gradient = A.T @ (p - y) + penalty * w
        hessian = (A * (p * (1 - p))[:, None]).T @ A + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Probabilities from a design matrix and fitted weights."""
    return 1.0 / (1.0 + np.exp(-(w[0] + X @ w[1:])))


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC by the rank-sum (Mann-Whitney) statistic, ties averaged."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = y.astype(bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fold_of(group: str, folds: int) -> int:
    """A volume's cross-validation fold: stable across runs."""
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % folds


def model_document(columns: dict, names: list[str], w: np.ndarray, info: dict) -> dict:
    """The JSON mapsnap.confidence.p_good reads: per named column its transform and coefficients."""
    out: dict = {**info, "intercept": float(w[0]), "columns": {}}
    k = len(columns)
    indicator_index = k
    for j, name in enumerate(names):
        column = columns[j]
        entry = {
            "fill": column["fill"],
            "mean": column["mean"],
            "scale": column["scale"],
            "coef": float(w[1 + j]),
        }
        if column["indicator"]:
            entry["missing_coef"] = float(w[1 + indicator_index])
            indicator_index += 1
        out["columns"][name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("volumes", type=Path, nargs="+")
    parser.add_argument("--iiif", default="mapsnap.iiif.json")
    parser.add_argument("--version", required=True, help="Recorded with every score.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw, labels, groups, panels = [], [], [], []
    names: list[str] = []
    for volume in args.volumes:
        try:
            rows = volume_rows(volume, args.iiif)
        except (Exception, SystemExit) as error:  # noqa: BLE001 -- skip a broken volume, say so
            print(f"{volume.name}: skipped ({error})", file=sys.stderr)
            continue
        for features, rmse in rows:
            values, names = feature_vector(features)
            raw.append([np.nan if v is None else v for v in values])
            labels.append(rmse <= GOOD_FT)
            groups.append(volume.name)
            panels.append(bool(features["panel"]))
    X_raw = np.array(raw, dtype=float)
    y = np.array(labels, dtype=float)
    group_array = np.array(groups)
    panel_mask = np.array(panels)
    print(
        f"{len(y)} placed poses in {len(set(groups))} volumes; {y.mean():.1%} within {GOOD_FT:g} ft"
    )

    oof = np.zeros(len(y))
    fold = np.array([fold_of(g, args.folds) for g in group_array])
    for k in range(args.folds):
        test = fold == k
        X_train, columns = design(X_raw[~test])
        X_test, _ = design(X_raw[test], columns)
        oof[test] = predict(X_test, fit_logistic(X_train, y[~test], l2=args.l2))
    metrics = {
        "auc": round(auc(y, oof), 4),
        "auc_pages": round(auc(y[~panel_mask], oof[~panel_mask]), 4),
        "auc_panels": round(auc(y[panel_mask], oof[panel_mask]), 4),
    }
    print(
        f"grouped {args.folds}-fold AUC {metrics['auc']} (pages {metrics['auc_pages']}, panels {metrics['auc_panels']})"
    )
    for lo in np.arange(0.0, 1.0, 0.1):
        band = (oof >= lo) & (oof < lo + 0.1)
        if band.sum() >= 30:
            print(
                f"  P {lo:.1f}-{lo + 0.1:.1f}: n={band.sum():5d}  observed good {y[band].mean():5.1%}"
            )

    X_all, columns = design(X_raw)
    w = fit_logistic(X_all, y, l2=args.l2)
    info = {
        "version": args.version,
        "target": f"rmse <= {GOOD_FT:g} ft against OIM truth",
        "trained": datetime.now(UTC).strftime("%Y-%m-%d"),
        "n_poses": len(y),
        "n_volumes": len(set(groups)),
        "cv": metrics,
    }
    args.out.write_text(
        json.dumps(model_document(columns, names, w, info), indent=1) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
