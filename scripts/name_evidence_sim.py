"""Offline re-ranking of cached snap candidates under signed name evidence.

Issue #375: name_alignment's reward-only score cannot penalize a pose whose
labels match nothing, so a grid alias with strong P(road) evidence can outrank
the truth. Every candidate's name block is logged in
``artifacts/osm_snap/candidates.jsonl``, so the change can be measured without
re-running snap: re-rank each page's cached candidates with the signed term and
compare the new winner's grid rmse against the old one's.

Grid rmse is the cheap diagnostic (a 7x7 lattice against one truth affine); it
diverges from region-graded rmse on split panels, so treat the output as a
ranking signal and judge production changes on real `mapsnap compare` runs.

    uv run python scripts/name_evidence_sim.py [--miss 0.5] [--min-labels 3]
"""

import argparse
import glob
import json
import statistics
from collections import Counter

from mapsnap.osm_snap import name_evidence

GOOD_FT = 50.0
BAD_FT = 500.0


def candidate_rows(pattern: str) -> list[dict]:
    """Every cached candidate with both a grid rmse and a name block."""
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        volume = path.split("/")[1]
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for candidate in record.get("candidates") or []:
                    name = candidate.get("name") or {}
                    if candidate.get("rmse_ft") is None or name.get("n_labels") is None:
                        continue
                    rows.append(
                        {
                            "volume": volume,
                            "page": record["target"],
                            "rmse_ft": candidate["rmse_ft"],
                            "select_score": candidate.get("select_score"),
                            "score": name.get("score"),
                            "n_labels": name["n_labels"],
                            "n_hits": name.get("n_hits") or 0,
                        }
                    )
    return rows


def report_discrimination(rows: list[dict], min_labels: int) -> None:
    """How well 'no name hits' separates accurate poses from wrong ones."""
    buckets: Counter[tuple[str, str]] = Counter()
    fractions: dict[str, list[float]] = {"good": [], "bad": []}
    for row in rows:
        if row["n_labels"] < min_labels:
            continue
        quality = (
            "good"
            if row["rmse_ft"] <= GOOD_FT
            else "bad"
            if row["rmse_ft"] >= BAD_FT
            else "mid"
        )
        buckets[(quality, "zero" if row["n_hits"] == 0 else "some")] += 1
        if quality in fractions:
            fractions[quality].append(row["n_hits"] / row["n_labels"])
    print(f"candidates with >={min_labels} eligible labels:")
    for quality in ("good", "bad", "mid"):
        zero = buckets[(quality, "zero")]
        total = zero + buckets[(quality, "some")]
        if total:
            print(
                f"  {quality:4s}: {total:6d} candidates, zero hits {zero:5d} ({100 * zero / total:5.1f}%)"
            )
    for quality, values in fractions.items():
        if values:
            print(f"  {quality} hit fraction: median {statistics.median(values):.2f}")


def report_reranking(rows: list[dict], miss: float, min_labels: int) -> None:
    """Where the top-ranked candidate changes, and whether it gets closer."""
    pages: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["select_score"] is not None:
            pages.setdefault((row["volume"], row["page"]), []).append(row)

    def signed(row: dict) -> float:
        return (
            row["select_score"]
            - (row["score"] or 0.0)
            + name_evidence(row["score"] or 0.0, row["n_labels"], row["n_hits"])
        )

    flips, better, worse, rescues, losses = [], 0, 0, [], []
    for (volume, page), candidates in sorted(pages.items()):
        if len(candidates) < 2:
            continue
        old = max(candidates, key=lambda c: c["select_score"])
        new = max(candidates, key=signed)
        if new is old:
            continue
        flips.append((volume, page, old["rmse_ft"], new["rmse_ft"]))
        if new["rmse_ft"] < old["rmse_ft"] * 0.8:
            better += 1
        elif new["rmse_ft"] > old["rmse_ft"] * 1.25:
            worse += 1
        if old["rmse_ft"] >= 200 and new["rmse_ft"] <= 50:
            rescues.append((volume, page, old["rmse_ft"], new["rmse_ft"]))
        if old["rmse_ft"] <= 50 and new["rmse_ft"] >= 200:
            losses.append((volume, page, old["rmse_ft"], new["rmse_ft"]))

    print(
        f"\nmiss cost {miss}, label floor {min_labels}: {len(flips)} pages change winner"
    )
    print(f"  materially better (>=20% closer): {better}")
    print(f"  materially worse  (>=25% farther): {worse}")
    print(f"  disaster -> good: {len(rescues)}   good -> disaster: {len(losses)}")
    for volume, page, before, after in sorted(rescues, key=lambda r: -r[2]):
        print(f"    {volume:28s} {page:9s} {before:9.1f} -> {after:7.1f} ft")
    for volume, page, before, after in losses:
        print(f"    LOSS {volume:24s} {page:9s} {before:9.1f} -> {after:7.1f} ft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss", type=float, default=None, help="override W_NAME_MISS")
    parser.add_argument(
        "--min-labels", type=int, default=None, help="override NAME_MISS_MIN_LABELS"
    )
    parser.add_argument(
        "--candidates",
        default="data/*/artifacts/osm_snap/candidates.jsonl",
        help="glob of cached candidate files",
    )
    args = parser.parse_args()
    if args.miss is not None or args.min_labels is not None:
        import mapsnap.osm_snap as snap

        if args.miss is not None:
            snap.W_NAME_MISS = args.miss
        if args.min_labels is not None:
            snap.NAME_MISS_MIN_LABELS = args.min_labels
    from mapsnap.osm_snap import NAME_MISS_MIN_LABELS, W_NAME_MISS

    rows = candidate_rows(args.candidates)
    print(f"{len(rows)} cached candidates with name blocks")
    report_discrimination(rows, NAME_MISS_MIN_LABELS)
    report_reranking(rows, W_NAME_MISS, NAME_MISS_MIN_LABELS)


if __name__ == "__main__":
    main()
