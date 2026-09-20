#!/usr/bin/env python
"""What a finished Batch array cost, how it was interrupted, and what it implies.

    scripts/batch/collect-run.py <array-job-id> <items.txt> \\
        --instances instances.tsv --out run-report.json

Run it soon after the array drains. Two of its three inputs expire: Batch
drops a child's attempt history 24 hours after the child completes, and a
terminated instance leaves ``describe-instances`` within the hour, which is
why ``snapshot-instances.sh`` has to run *alongside* the job rather than after
it. Without ``--instances`` the throughput and interrupt numbers still come
out; the cost does not.

The corpus projection uses the run's own measured curve applied to the
mirror's size distribution, not its per-item average: a run's items are rarely
a fair sample, and the 200-item pilot's were 2.4x the corpus average, which
made a per-item projection overstate the bill by the same factor.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mapsnap.batch_report import (
    empirical_seconds,
    expected_rework,
    fit_cost_model,
    format_dollars,
    interrupt_rate,
    interruptions_per_second,
    read_fleet_usage,
    utilisation,
)
from mapsnap.loc_fit import count_sheets

REGION = "us-west-2"
SUMMARY = "items fitted"


def aws(*args: str) -> dict:
    """An aws CLI call that returns JSON."""
    result = subprocess.run(
        ["aws", *args, "--region", REGION, "--output", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def describe_children(parent: str, count: int) -> list[dict]:
    """Every array child's record, in batches of 100 (the API's limit)."""
    ids = [f"{parent}:{index}" for index in range(count)]
    jobs: list[dict] = []
    for start in range(0, len(ids), 100):
        jobs += aws("batch", "describe-jobs", "--jobs", *ids[start : start + 100]).get(
            "jobs", []
        )
    return jobs


def log_tail(stream: str, limit: int = 40) -> str:
    """The last lines of a child's log, or '' when it has aged out."""
    try:
        events = aws(
            "logs",
            "get-log-events",
            "--log-group-name",
            "/aws/batch/job",
            "--log-stream-name",
            stream,
            "--limit",
            str(limit),
        )
    except subprocess.CalledProcessError:
        return ""
    return "\n".join(event["message"] for event in events.get("events", []))


def spot_price(kind: str, zone: str) -> float:
    """The latest spot price for an instance type in a zone, dollars per hour."""
    history = aws(
        "ec2",
        "describe-spot-price-history",
        "--instance-types",
        kind,
        "--availability-zone",
        zone,
        "--product-descriptions",
        "Linux/UNIX",
        "--max-items",
        "1",
    )
    entries = history.get("SpotPriceHistory") or []
    return float(entries[0]["SpotPrice"]) if entries else 0.0


def pages_and_rss(text: str) -> tuple[int | None, int | None]:
    """The pages fitted and peak stage RSS from loc-fit's own summary lines.

    A chunked child prints one summary per item, so the pages add up and the
    peak is the highest any of them reached.
    """
    pages = rss = None
    for line in text.splitlines():
        if SUMMARY not in line:
            continue
        for token, follows in (("pages in", "pages"), ("peak stage RSS", "rss")):
            if token not in line:
                continue
            words = line.split()
            for index, word in enumerate(words):
                if (
                    follows == "pages"
                    and word == "pages"
                    and index
                    and words[index - 1].isdigit()
                ):
                    pages = (pages or 0) + int(words[index - 1])
                if (
                    follows == "rss"
                    and word == "RSS"
                    and index + 1 < len(words)
                    and words[index + 1].isdigit()
                ):
                    rss = max(rss or 0, int(words[index + 1]))
    return pages, rss


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job", help="The array job id")
    parser.add_argument("items", type=Path, help="The items list the job ran")
    parser.add_argument(
        "--instances", type=Path, help="snapshot-instances.sh output, for the cost"
    )
    parser.add_argument("--out", type=Path, default=Path("run-report.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv",
        help="The mirror's manifest, for the corpus projection (default: %(default)s)",
    )
    parser.add_argument("--items-per-job", type=int, default=1, metavar="N")
    args = parser.parse_args()

    names = [
        line.strip() for line in args.items.read_text().splitlines() if line.strip()
    ]
    children = describe_children(args.job, -(-len(names) // args.items_per_job))
    rows = []
    for job in children:
        index = job["arrayProperties"]["index"]
        started, stopped = job.get("startedAt"), job.get("stoppedAt")
        stream = (job.get("container") or {}).get("logStreamName")
        row = {
            "index": index,
            "items": names[
                index * args.items_per_job : (index + 1) * args.items_per_job
            ],
            "status": job["status"],
            "exit_code": (job.get("container") or {}).get("exitCode"),
            "attempt_reasons": [
                a.get("statusReason", "") for a in job.get("attempts") or []
            ],
            "seconds": (stopped - started) / 1000 if started and stopped else None,
            "log_stream": stream,
        }
        if job["status"] == "SUCCEEDED" and stream:
            row["pages"], row["peak_rss_mb"] = pages_and_rss(log_tail(stream))
        rows.append(row)

    done = [r for r in rows if r["status"] == "SUCCEEDED"]
    failed = [r for r in rows if r["status"] == "FAILED"]
    durations = [r["seconds"] for r in rows if r["seconds"]]
    job_hours = sum(durations) / 3600
    pages = sum(r.get("pages") or 0 for r in done)
    hit, rate = interrupt_rate([r["attempt_reasons"] for r in rows])

    cost = {}
    total_cost = 0.0
    fleet = (
        read_fleet_usage(args.instances)
        if args.instances and args.instances.exists()
        else None
    )
    if fleet:
        for (kind, zone), seconds in sorted(fleet.seconds_by_shape.items()):
            price = spot_price(kind, zone)
            amount = seconds / 3600 * price
            cost[f"{kind} {zone}"] = {
                "instance_hours": round(seconds / 3600, 2),
                "spot_price": price,
                "cost": round(amount, 2),
            }
            total_cost += amount

    report = {
        "job": args.job,
        "children": len(rows),
        "items": len(names),
        "items_per_job": args.items_per_job,
        "succeeded": len(done),
        "failed": len(failed),
        "exit_codes": dict(Counter(r["exit_code"] for r in failed)),
        "job_hours": round(job_hours, 2),
        "pages_fitted": pages,
        "peak_rss_mb_max": max((r.get("peak_rss_mb") or 0 for r in done), default=0),
        "spot_interrupted_children": hit,
        "spot_interrupt_rate": round(rate, 4),
        "cost_by_instance": cost,
        "total_cost": round(total_cost, 4),
        "rows": rows,
    }
    if fleet:
        report["fleet_instances"] = fleet.instances
        report["fleet_instance_hours"] = round(fleet.instance_hours, 2)
        report["vcpu_utilisation"] = round(utilisation(job_hours, fleet), 3)
        report["spot_reclaimed_instances"] = fleet.spot_reclaimed
    if pages:
        report["cost_per_page"] = round(total_cost / pages, 6)

    print(
        f"=== {args.job}: {len(done)} of {len(rows)} children, {len(names):,} items ==="
    )
    print(f"  job time {job_hours:.1f} h, {pages:,} pages fitted")
    if durations:
        ordered = sorted(durations)
        print(
            f"  child minutes: median {ordered[len(ordered) // 2] / 60:.0f}, longest {ordered[-1] / 60:.0f}"
        )
    print(f"  peak stage RSS across items: {report['peak_rss_mb_max']} MB")
    print(
        f"  spot: {hit} children interrupted ({rate:.1%})"
        + (f", {fleet.spot_reclaimed} instances reclaimed" if fleet else "")
    )
    if failed:
        print(
            f"  failed: {len(failed)} — exit codes {report['exit_codes']} (scripts/batch/retry-list.sh sorts them)"
        )
    if fleet:
        for label, entry in cost.items():
            print(
                f"    {label}: {entry['instance_hours']} h x ${entry['spot_price']}/h = {format_dollars(entry['cost'])}"
            )
        print(
            f"  cost {format_dollars(total_cost)} over {fleet.instances} instances; "
            f"vCPU utilisation {report['vcpu_utilisation']:.0%}"
        )
        if pages:
            print(f"  per page {format_dollars(total_cost / pages)}")

    samples = [
        (r["pages"], r["seconds"]) for r in done if r.get("pages") and r.get("seconds")
    ]
    if samples and fleet and job_hours > 0:
        per_job_hour = total_cost / job_hours
        model = fit_cost_model(samples, per_job_hour)
        sheets = count_sheets(args.manifest) if args.manifest.exists() else {}
        if model and sheets:
            curve = empirical_seconds(samples)
            corpus_seconds = sum(curve(n) for n in sheets.values())
            # A child holds items_per_job items, so its own fixed cost is paid
            # once for the chunk rather than once per item.
            corpus_hours = corpus_seconds / 3600
            report["model"] = {
                "fixed_seconds_per_item": round(model.fixed_seconds_per_item, 1),
                "seconds_per_page": round(model.seconds_per_page, 2),
                "dollars_per_job_hour": round(per_job_hour, 4),
            }
            report["projection"] = {
                "corpus_items": len(sheets),
                "corpus_sheets": sum(sheets.values()),
                "job_hours": round(corpus_hours),
                "dollars": round(corpus_hours * per_job_hour),
            }
            print(
                f"\n  model: {model.fixed_seconds_per_item:.0f} s fixed per item + "
                f"{model.seconds_per_page:.1f} s per page, at {format_dollars(per_job_hour)}/job-hour"
            )
            print(
                f"  corpus ({len(sheets):,} items / {sum(sheets.values()):,} sheets): "
                f"{corpus_hours:,.0f} job-hours, {format_dollars(corpus_hours * per_job_hour)}"
            )
            for slots in (128, 256):
                print(
                    f"    {corpus_hours / slots:.0f} h of wall clock at {slots} concurrent children"
                )
        if samples:
            hazard = interruptions_per_second(rate, durations)
            print(
                f"  rework at this interrupt rate: {expected_rework(durations, hazard) / 3600:.1f} job-hours "
                f"({expected_rework(durations, hazard) / max(sum(durations), 1):.1%} of the run)"
            )

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
