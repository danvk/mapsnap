#!/usr/bin/env python
"""Split the mirror into a random sample and everything else.

    scripts/batch/sample-items.py --size 1000 --out-prefix corpus

writes `corpus-sample.txt` and `corpus-rest.txt`, which together are every
item in the manifest exactly once. Run the sample first to check that the cost
and the failure rate are what the pilot predicted; then run the rest **under
the same run tag**, and the two together are a complete corpus run. Nothing
about a run tag cares that it was filled in two passes, and an item already
published under it is skipped for the price of one listing, so the halves can
even overlap without harm.

The draw is seeded, so the same `--seed` always gives the same split: the
sample can be reproduced, and the rest can be regenerated months later without
keeping the file.

Sample sizes worth knowing, against a 10,000-child array cap:

    1,000 items at 8 per child =   125 children, about 3% of the corpus
   35,159 items at 8 per child = 4,395 children, the whole mirror

Balance each list before submitting -- the draw is random, so its chunks are
as lopsided as any other unplanned list:

    scripts/batch/plan-items.py corpus-sample.txt --per-job 8 --out planned.txt
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mapsnap.batch_report import summarise_sizes
from mapsnap.loc_fit import count_sheets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Items in the sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260920, help="Draw seed (default: %(default)s)"
    )
    parser.add_argument(
        "--out-prefix",
        default="corpus",
        help="Prefix for the two lists (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv",
        help="The mirror's per-sheet manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help="An items list to leave out of both halves (already-run items, say)",
    )
    args = parser.parse_args()

    sheets = count_sheets(args.manifest)
    names = sorted(sheets)
    if args.exclude:
        skip = {
            line.strip()
            for line in args.exclude.read_text().splitlines()
            if line.strip()
        }
        names = [name for name in names if name not in skip]
        print(f"excluded {len(skip):,} items named in {args.exclude}", file=sys.stderr)
    if args.size >= len(names):
        sys.exit(
            f"--size {args.size:,} is not smaller than the {len(names):,} items available"
        )

    sample = sorted(random.Random(args.seed).sample(names, args.size))
    chosen = set(sample)
    rest = [name for name in names if name not in chosen]

    sample_path = Path(f"{args.out_prefix}-sample.txt")
    rest_path = Path(f"{args.out_prefix}-rest.txt")
    sample_path.write_text("\n".join(sample) + "\n")
    rest_path.write_text("\n".join(rest) + "\n")

    assert len(sample) + len(rest) == len(names), "the split lost or duplicated items"
    print(
        f"{sample_path}: {summarise_sizes([sheets[n] for n in sample])}",
        file=sys.stderr,
    )
    print(f"{rest_path}: {summarise_sizes([sheets[n] for n in rest])}", file=sys.stderr)
    print(f"all items:  {summarise_sizes([sheets[n] for n in names])}", file=sys.stderr)
    sample_mean = sum(sheets[n] for n in sample) / len(sample)
    overall_mean = sum(sheets[n] for n in names) / len(names)
    print(
        f"\nthe sample averages {sample_mean / overall_mean:.2f}x the typical item; "
        "anything near 1.00 projects honestly, and the 200-item pilot was 2.37x",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
