"""Train snap's accept model (models/snap_accept.json) from graded candidates (#650).

    train_snap_accept.py GRADED_JSONL [--exclude ITEMS_TXT] [--output PATH]

GRADED_JSONL is grade_snap_candidates.py's output. Two logistic models, on
the features mapsnap.snap_accept computes from each record:

  - rescue: pages with no incumbent; target = the top candidate is within
    25 ft of truth.
  - replace: fitted pages whose top candidate disagrees with the incumbent by
    ARBITRATE_MIN_DISAGREE_FT or more; target = publishing it instead of the
    incumbent improves the page (good gained or disaster removed, without the
    reverse).

Each model's threshold maximizes, over out-of-fold predictions (folds by
item), the net pages it would change: good minus disaster for rescues, gains
minus losses for replacements.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from train_confidence import auc, fit_logistic, fold_of, predict

from mapsnap.snap_accept import (
    ACCEPT_MODEL_PATH,
    replace_terms,
    rescue_terms,
    scorable,
)
from mapsnap.snap_volume import (
    ARBITRATE_MIN_DISAGREE_FT,
    RESCUE_STATES,
    incumbent_disagreement_ft,
)

GOOD_FT = 25.0
DISASTER_FT = 200.0
FOLDS = 5


def band(rmse_ft: float) -> tuple[int, int]:
    """(good, disaster) for a graded pose."""
    return int(rmse_ft <= GOOD_FT), int(rmse_ft >= DISASTER_FT)


def rescue_example(record: dict) -> tuple[dict[str, float], float] | None:
    """(terms, value of accepting) for a rescue page; value is +1 good, -1 disaster, 0 else."""
    if record.get("fit_state") not in RESCUE_STATES or not scorable(record):
        return None
    rmse = record["candidates"][0].get("rmse_ft")
    if rmse is None:
        return None
    good, disaster = band(rmse)
    return rescue_terms(record), float(good - disaster)


def replace_example(record: dict) -> tuple[dict[str, float], float] | None:
    """(terms, value of replacing) for a fitted page whose top candidate disagrees with it."""
    incumbent = record.get("incumbent") or {}
    if record.get("fit_state") != "fitted" or not scorable(record):
        return None
    if not incumbent.get("world_affine") or incumbent.get("rmse_ft") is None:
        return None
    rmse = record["candidates"][0].get("rmse_ft")
    disagreement = incumbent_disagreement_ft(record)
    if rmse is None or disagreement is None or disagreement < ARBITRATE_MIN_DISAGREE_FT:
        return None
    good_c, disaster_c = band(rmse)
    good_i, disaster_i = band(incumbent["rmse_ft"])
    return replace_terms(record), float((good_c - good_i) - (disaster_c - disaster_i))


def standardize(
    rows: list[dict[str, float]], columns: dict | None = None
) -> tuple[np.ndarray, dict]:
    """The design matrix and each term's mean and scale (fitted when ``columns`` is None)."""
    names = list(rows[0])
    raw = np.array([[row[name] for name in names] for row in rows])
    if columns is None:
        columns = {
            name: {
                "mean": float(raw[:, j].mean()),
                "scale": float(raw[:, j].std()) or 1.0,
            }
            for j, name in enumerate(names)
        }
    X = np.column_stack(
        [
            (raw[:, j] - columns[name]["mean"]) / columns[name]["scale"]
            for j, name in enumerate(names)
        ]
    )
    return X, columns


def best_threshold(
    probabilities: np.ndarray, values: np.ndarray
) -> tuple[float, float]:
    """(threshold, net value) accepting every example at or above the threshold."""
    order = np.argsort(-probabilities)
    totals = np.cumsum(values[order])
    best = int(np.argmax(totals))
    if totals[best] <= 0:
        return 1.0, 0.0
    return float(probabilities[order][best]), float(totals[best])


def train(examples: list[tuple[dict[str, float], float, str]], label: str) -> dict:
    """Fit one model; report cross-validated AUC and net, and pick its threshold."""
    terms = [t for t, _, _ in examples]
    values = np.array([v for _, v, _ in examples])
    groups = [g for _, _, g in examples]
    y = (values > 0).astype(float)
    folds = np.array([fold_of(group, FOLDS) for group in groups])
    out_of_fold = np.zeros(len(examples))
    for fold in range(FOLDS):
        test = folds == fold
        X_train, columns = standardize([t for t, keep in zip(terms, ~test) if keep])
        w = fit_logistic(X_train, y[~test])
        X_test, _ = standardize([t for t, keep in zip(terms, test) if keep], columns)
        out_of_fold[test] = predict(X_test, w)
    threshold, net = best_threshold(out_of_fold, values)
    X, columns = standardize(terms)
    w = fit_logistic(X, y)
    for j, name in enumerate(columns):
        columns[name]["coef"] = float(w[1 + j])
    accepted = out_of_fold >= threshold
    print(
        f"{label}: {len(examples)} pages, AUC {auc(y, out_of_fold):.3f}; "
        f"threshold {threshold:.3f} accepts {int(accepted.sum())}, "
        f"+{int((values[accepted] > 0).sum())} / -{int((values[accepted] < 0).sum())}, "
        f"net {net:+.0f} (out of fold)"
    )
    return {
        "intercept": float(w[0]),
        "threshold": threshold,
        "cv": {"auc": auc(y, out_of_fold), "net": net, "n": len(examples)},
        "columns": columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("graded", type=Path)
    parser.add_argument(
        "--exclude",
        type=Path,
        help="Items (one per line) to leave out, e.g. an A/B's held-out volumes.",
    )
    parser.add_argument("--output", type=Path, default=ACCEPT_MODEL_PATH)
    parser.add_argument("--version", default="corpus-v2")
    args = parser.parse_args()

    excluded = set()
    if args.exclude:
        excluded = {line.strip() for line in args.exclude.open() if line.strip()}
    rescues: list[tuple[dict[str, float], float, str]] = []
    replaces: list[tuple[dict[str, float], float, str]] = []
    items = set()
    for line in args.graded.open():
        record = json.loads(line)
        if record["item"] in excluded:
            continue
        items.add(record["item"])
        if (example := rescue_example(record)) is not None:
            rescues.append((*example, record["item"]))
        elif (example := replace_example(record)) is not None:
            replaces.append((*example, record["item"]))
    document = {
        "version": args.version,
        "trained": datetime.now(UTC).strftime("%Y-%m-%d"),
        "n_items": len(items),
        "excluded_items": len(excluded),
        "rescue": train(rescues, "rescue"),
        "replace": train(replaces, "replace"),
    }
    args.output.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
