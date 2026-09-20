"""What a Batch run cost, how much of the fleet it used, and what that implies.

Three numbers decide whether the corpus run is affordable, and each has to be
read from a different place before it expires:

* **cost** from the instance-seconds the fleet actually ran, priced at the spot
  rate for that instance type and zone. Cost Explorer lags a day and cannot
  separate one job from the rest of the account, so the fleet is sampled while
  the job runs (``scripts/batch/snapshot-instances.sh``): a terminated instance
  leaves ``describe-instances`` within the hour.
* **the spot interrupt rate** from the children's attempt history. Batch keeps
  that for 24 hours after a child completes.
* **throughput** from the children's own start and stop stamps, which is also
  where the fixed-plus-per-page model comes from.

The projection is the part worth being careful about. A run's items are rarely
a fair sample of the mirror -- the 200-item pilot averaged 29.7 sheets against
the corpus's 12.5 -- so projecting per *item* overstated the corpus by 2.4x.
:func:`fit_cost_model` separates the per-item cost from the per-page cost so
the projection can be applied to the corpus's own size distribution.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# vCPU and GiB for the shapes the compute environment asks for. Used to work
# out how much of what we rented the jobs actually occupied.
INSTANCE_SHAPES: dict[str, tuple[int, int]] = {
    "m5.2xlarge": (8, 32),
    "m5a.2xlarge": (8, 32),
    "m6a.2xlarge": (8, 32),
    "m6i.2xlarge": (8, 32),
    "r5.2xlarge": (8, 64),
    "r6i.2xlarge": (8, 64),
    "c5.2xlarge": (8, 16),
    "c6a.2xlarge": (8, 16),
    "c6i.2xlarge": (8, 16),
}

SNAPSHOT_COLUMNS = ("ts", "instance", "type", "az", "state", "launch", "transition")


@dataclass(frozen=True)
class FleetUsage:
    """What the fleet supplied, and how much of it the jobs occupied."""

    seconds_by_shape: dict[tuple[str, str], float]
    end_reasons: dict[str, str]
    instances: int

    @property
    def instance_hours(self) -> float:
        return sum(self.seconds_by_shape.values()) / 3600

    @property
    def vcpu_hours(self) -> float:
        return sum(
            seconds / 3600 * INSTANCE_SHAPES.get(kind, (8, 32))[0]
            for (kind, _), seconds in self.seconds_by_shape.items()
        )

    @property
    def spot_reclaimed(self) -> int:
        """Instances EC2 took back, as opposed to ones Batch scaled down."""
        return sum(1 for reason in self.end_reasons.values() if "Spot" in reason)


@dataclass(frozen=True)
class CostModel:
    """A run's cost as a fixed charge per item plus a marginal charge per page."""

    fixed_seconds_per_item: float
    seconds_per_page: float
    dollars_per_job_hour: float
    samples: int

    def seconds_for(self, items: int, pages: int) -> float:
        """Job-seconds this model predicts for a corpus of that shape."""
        return items * self.fixed_seconds_per_item + pages * self.seconds_per_page

    def project(self, items: int, pages: int) -> dict[str, float]:
        """Job-hours, vCPU-hours and dollars for a corpus of ``items``/``pages``."""
        seconds = self.seconds_for(items, pages)
        return {
            "job_hours": seconds / 3600,
            "vcpu_hours": seconds / 3600 * 2,
            "dollars": seconds / 3600 * self.dollars_per_job_hour,
        }


def parse_instant(text: str) -> datetime:
    """A UTC datetime from either snapshot stamp format (``Z`` or ``+00:00``)."""
    cleaned = text.strip().replace("Z", "+00:00")
    moment = datetime.fromisoformat(cleaned)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def read_fleet_usage(path: Path) -> FleetUsage:
    """Instance-seconds per (type, zone) from a snapshot file, with end reasons.

    An instance runs from its launch time to the last snapshot that still saw
    it running; the poller's period is the resolution, which is well under a
    percent of a multi-hour run. A minute is the floor, since that is how spot
    is billed.
    """
    first: dict[str, datetime] = {}
    last: dict[str, datetime] = {}
    shape: dict[str, tuple[str, str]] = {}
    reasons: dict[str, str] = {}
    lines = path.read_text().splitlines()
    for line in lines[1:]:  # the header names SNAPSHOT_COLUMNS
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        stamp, instance, kind, zone, state, launch = fields[:6]
        transition = fields[6] if len(fields) > 6 else ""
        seen = parse_instant(stamp)
        shape[instance] = (kind, zone)
        first.setdefault(instance, parse_instant(launch))
        if state == "running":
            last[instance] = max(last.get(instance, seen), seen)
        if transition.strip():
            reasons[instance] = transition.strip()
    seconds: dict[tuple[str, str], float] = {}
    for instance, started in first.items():
        alive = (last.get(instance, started) - started).total_seconds()
        key = shape[instance]
        seconds[key] = seconds.get(key, 0.0) + max(alive, 60.0)
    return FleetUsage(
        seconds_by_shape=seconds, end_reasons=reasons, instances=len(first)
    )


def fit_cost_model(
    samples: list[tuple[int, float]], dollars_per_job_hour: float
) -> CostModel | None:
    """Least-squares fit of seconds = fixed + per_page * pages over finished items.

    Returns None below ten samples, where the intercept is noise. The intercept
    is what a chunked run amortises and the slope is what it cannot, so keeping
    them apart is what makes a projection onto a differently-shaped corpus
    honest.
    """
    usable = [
        (pages, seconds) for pages, seconds in samples if pages > 0 and seconds > 0
    ]
    if len(usable) < 10:
        return None
    count = len(usable)
    sum_x = sum(pages for pages, _ in usable)
    sum_y = sum(seconds for _, seconds in usable)
    sum_xx = sum(pages * pages for pages, _ in usable)
    sum_xy = sum(pages * seconds for pages, seconds in usable)
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0:  # every item the same size: no slope to find
        return None
    per_page = (count * sum_xy - sum_x * sum_y) / denominator
    fixed = (sum_y - per_page * sum_x) / count
    return CostModel(
        fixed_seconds_per_item=fixed,
        seconds_per_page=per_page,
        dollars_per_job_hour=dollars_per_job_hour,
        samples=count,
    )


def empirical_seconds(samples: list[tuple[int, float]]):
    """A seconds-for-this-many-sheets function built from a run's own items.

    The linear model is the right shape for separating fixed from marginal
    cost, but per-page cost is not actually constant, so projecting over a
    corpus of mostly small volumes does better with the measured curve: the
    median of the items nearest that size.

    Past the largest item the run measured, it scales the largest measurement
    in proportion rather than reaching for the nearest neighbours -- those are
    every item at once, and their median is a small volume's time, which would
    price a 500-sheet volume like a five-sheet one.
    """
    pairs = sorted(samples)
    if not pairs:
        raise ValueError("no samples to build a curve from")
    largest_pages = pairs[-1][0]
    largest_seconds = statistics.median([s for p, s in pairs if p == largest_pages])

    def seconds_for(pages: int) -> float:
        if pages > largest_pages:
            return largest_seconds * pages / largest_pages
        window = max(2, pages * 0.25)
        near = [s for p, s in pairs if abs(p - pages) <= window]
        if not near:
            near = [s for _, s in sorted(pairs, key=lambda ps: abs(ps[0] - pages))[:5]]
        return statistics.median(near)

    return seconds_for


def interrupt_rate(attempt_reasons: list[list[str]]) -> tuple[int, float]:
    """Children whose host was taken back, and that as a share of all children.

    A reclamation is only visible here: the instance is gone from EC2 within
    the hour, but Batch remembers why the attempt ended.
    """
    if not attempt_reasons:
        return 0, 0.0
    hit = sum(
        1
        for reasons in attempt_reasons
        if any("host ec2" in reason.lower() for reason in reasons)
    )
    return hit, hit / len(attempt_reasons)


def interruptions_per_second(rate_per_child: float, durations: list[float]) -> float:
    """Convert a measured share-of-children-interrupted into a hazard per second.

    Spot takes back *hosts*, not children, so the underlying rate is per unit
    of running time; a run only ever reports the share of its children that
    were hit. Dividing by the mean child length recovers the rate the hosts
    were actually being reclaimed at, which is what lets two chunkings with
    different child counts be compared.
    """
    if not durations:
        return 0.0
    mean_duration = sum(durations) / len(durations)
    return rate_per_child / mean_duration if mean_duration > 0 else 0.0


def expected_rework(durations: list[float], per_second_rate: float) -> float:
    """Seconds a run expects to redo, at a constant per-second interruption hazard.

    A child running for ``d`` seconds is exposed for all of them, so its chance
    of being hit goes as ``d`` and the work lost averages ``d / 2``: the total
    goes as the sum of the *squares*. That is why one ten-hour child is far
    worse than six ninety-minute ones holding the same work, and why balancing
    the chunks is worth more than the makespan alone suggests.
    """
    return per_second_rate / 2 * sum(d * d for d in durations)


def utilisation(job_hours: float, fleet: FleetUsage) -> float:
    """Share of the vCPU-hours we rented that the jobs actually occupied.

    Each job asks for two vCPUs. Well under 1 means the fleet's shapes cannot
    pack the job's memory request, which is what made the pilot cost twice what
    it should have.
    """
    supplied = fleet.vcpu_hours
    return (job_hours * 2 / supplied) if supplied > 0 else 0.0


def format_dollars(amount: float) -> str:
    """A dollar amount at the precision the number deserves."""
    if amount >= 100:
        return f"${amount:,.0f}"
    if amount >= 1:
        return f"${amount:,.2f}"
    return f"${amount:.5f}".rstrip("0")


def summarise_sizes(sheets: list[int]) -> str:
    """A one-line description of how big a set of items is."""
    if not sheets:
        return "no items"
    ordered = sorted(sheets)
    return (
        f"{len(ordered):,} items, {sum(ordered):,} sheets, "
        f"mean {statistics.mean(ordered):.1f}, median {statistics.median(ordered):.0f}, "
        f"p90 {ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.9))]}"
    )
