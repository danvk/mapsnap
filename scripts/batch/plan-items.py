#!/usr/bin/env python
"""Reorder an items list so a chunked Batch array runs balanced children.

`submit.sh` slices the list by position, so reordering is all it takes: child
i runs lines [i*PER_JOB, (i+1)*PER_JOB). Without this the chunks follow item
id, which is uncorrelated with volume size, and one child ends up holding
several of the mirror's 150-sheet volumes.

    scripts/batch/plan-items.py corpus-items.txt --per-job 8 --out planned.txt
    PER_JOB=8 scripts/batch/submit.sh corpus-v1 planned.txt

Simulated over the whole mirror at 8 items a child and 128 slots: the longest
child falls from 10.0 h to 1.7 h, the makespan from 54 h to 49 h, and the work
a 4% spot-interrupt rate forces us to redo from 4.7% of the run to 2.0%.
"""

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mapsnap.loc_fit import (
    FIXED_COST_IN_SHEETS,
    balance_items,
    count_sheets,
)


def chunk_weights(names: list[str], sheets: dict[str, int], per_job: int) -> list[int]:
    """Each child's total weight, in sheets plus the per-item fixed cost."""
    weights = [sheets.get(name, 1) + FIXED_COST_IN_SHEETS for name in names]
    return [sum(weights[i : i + per_job]) for i in range(0, len(weights), per_job)]


def describe(
    label: str, names: list[str], sheets: dict[str, int], per_job: int
) -> None:
    """Print how lopsided the children of this ordering would be."""
    weights = sorted(chunk_weights(names, sheets, per_job))
    print(
        f"  {label:14} {len(weights):6,} children  mean {st.mean(weights):6.0f}  "
        f"p99 {weights[int(len(weights) * 0.99)]:6,}  worst {weights[-1]:6,} sheet-equivalents"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("items", type=Path, help="Items list, one id per line")
    parser.add_argument(
        "--per-job",
        type=int,
        default=8,
        metavar="N",
        help="Items per child (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv",
        help="The mirror's per-sheet manifest, for sheet counts (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, help="Where to write the reordered list (default: stdout)"
    )
    args = parser.parse_args()

    names = [
        line.strip() for line in args.items.read_text().splitlines() if line.strip()
    ]
    sheets = count_sheets(args.manifest)
    missing = [name for name in names if name not in sheets]
    if missing:
        print(
            f"{len(missing)} items are not in the manifest and are weighted as one sheet: {missing[:3]}",
            file=sys.stderr,
        )
    planned = balance_items(names, sheets, args.per_job)
    if sorted(planned) != sorted(names):
        sys.exit("the plan lost or duplicated items; refusing to write it")
    print(f"{len(names):,} items at {args.per_job} per child:", file=sys.stderr)
    describe("list order", names, sheets, args.per_job)
    describe("balanced", planned, sheets, args.per_job)
    text = "\n".join(planned) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
