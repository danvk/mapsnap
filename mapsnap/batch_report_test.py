import math

import pytest

from mapsnap.batch_report import (
    CostModel,
    empirical_seconds,
    expected_rework,
    fit_cost_model,
    format_dollars,
    interrupt_rate,
    interruptions_per_second,
    parse_instant,
    read_fleet_usage,
    summarise_sizes,
    utilisation,
)


def write_snapshots(path, rows: list[tuple[str, ...]]) -> None:
    """A snapshot file in the shape snapshot-instances.sh writes."""
    header = "ts\tinstance\ttype\taz\tstate\tlaunch\ttransition\n"
    path.write_text(header + "".join("\t".join(r) + "\n" for r in rows))


def test_read_fleet_usage_measures_each_instance_from_launch_to_last_seen(tmp_path):
    path = tmp_path / "instances.tsv"
    write_snapshots(
        path,
        [
            (
                "2026-09-19T22:00:00Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "running",
                "2026-09-19T21:50:00+00:00",
                "",
            ),
            (
                "2026-09-19T22:30:00Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "running",
                "2026-09-19T21:50:00+00:00",
                "",
            ),
            (
                "2026-09-19T22:30:00Z",
                "i-2",
                "r5.2xlarge",
                "us-west-2b",
                "running",
                "2026-09-19T22:20:00+00:00",
                "",
            ),
        ],
    )
    fleet = read_fleet_usage(path)
    assert fleet.instances == 2
    # i-1 ran 21:50 to 22:30 = 40 min; i-2 from 22:20 to 22:30 = 10 min.
    assert fleet.seconds_by_shape[("m5.2xlarge", "us-west-2a")] == 40 * 60
    assert fleet.seconds_by_shape[("r5.2xlarge", "us-west-2b")] == 10 * 60
    assert fleet.instance_hours == pytest.approx(50 / 60)
    assert fleet.vcpu_hours == pytest.approx(50 / 60 * 8)


def test_read_fleet_usage_bills_a_minute_minimum(tmp_path):
    path = tmp_path / "instances.tsv"
    write_snapshots(
        path,
        [
            (
                "2026-09-19T22:00:05Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "running",
                "2026-09-19T22:00:00+00:00",
                "",
            )
        ],
    )
    assert read_fleet_usage(path).seconds_by_shape[("m5.2xlarge", "us-west-2a")] == 60.0


def test_read_fleet_usage_records_why_an_instance_went_away(tmp_path):
    path = tmp_path / "instances.tsv"
    write_snapshots(
        path,
        [
            (
                "2026-09-19T22:00:00Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "running",
                "2026-09-19T21:50:00+00:00",
                "",
            ),
            (
                "2026-09-19T22:10:00Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "terminated",
                "2026-09-19T21:50:00+00:00",
                "Server.SpotInstanceTermination",
            ),
            (
                "2026-09-19T22:10:00Z",
                "i-2",
                "m5.2xlarge",
                "us-west-2a",
                "terminated",
                "2026-09-19T21:50:00+00:00",
                "User initiated",
            ),
        ],
    )
    fleet = read_fleet_usage(path)
    assert fleet.spot_reclaimed == 1
    assert fleet.end_reasons["i-1"] == "Server.SpotInstanceTermination"


def test_parse_instant_accepts_both_stamp_formats():
    assert parse_instant("2026-09-19T22:00:00Z") == parse_instant(
        "2026-09-19T22:00:00+00:00"
    )


def test_fit_cost_model_separates_the_fixed_cost_from_the_per_page_cost():
    # seconds = 100 + 40 * pages, exactly.
    samples = [(pages, 100.0 + 40.0 * pages) for pages in range(1, 30)]
    model = fit_cost_model(samples, dollars_per_job_hour=0.04)
    assert model is not None
    assert model.fixed_seconds_per_item == pytest.approx(100.0)
    assert model.seconds_per_page == pytest.approx(40.0)
    # A corpus of many small items is mostly fixed cost, which is the whole
    # reason for keeping the two apart.
    projected = model.project(items=1000, pages=4000)
    assert projected["job_hours"] == pytest.approx((1000 * 100 + 4000 * 40) / 3600)
    assert projected["dollars"] == pytest.approx(projected["job_hours"] * 0.04)


def test_fit_cost_model_declines_when_there_is_too_little_to_fit():
    assert fit_cost_model([(3, 200.0)] * 5, 0.04) is None
    # Every item the same size leaves the slope undetermined.
    assert fit_cost_model([(3, 200.0)] * 20, 0.04) is None


def test_empirical_seconds_follows_the_measured_curve():
    samples = [(1, 60.0), (1, 80.0), (2, 100.0), (50, 3000.0), (60, 3600.0)]
    curve = empirical_seconds(samples)
    assert curve(1) == pytest.approx(80.0)  # median of the items near one page
    assert curve(55) == pytest.approx(3300.0)
    # Past the largest item measured it scales that one, rather than taking a
    # median over every item, which would price 500 sheets like a handful.
    assert curve(120) == pytest.approx(3600.0 * 2)


def test_empirical_seconds_needs_samples():
    with pytest.raises(ValueError):
        empirical_seconds([])


def test_interrupt_rate_counts_only_reclaimed_hosts():
    reasons = [
        ["Host EC2 (instance i-abc) terminated.", "Essential container in task exited"],
        ["Essential container in task exited", "Essential container in task exited"],
        [],
        ["Essential container in task exited"],
    ]
    assert interrupt_rate(reasons) == (1, 0.25)
    assert interrupt_rate([]) == (0, 0.0)


def test_expected_rework_punishes_one_long_child_more_than_several_short_ones():
    # The same total work, split two ways, at the same hazard per second.
    hazard = 0.04 / (3600.0 * 8)  # a 4% chance for a child of the longer length
    long_child = expected_rework([3600.0 * 8], hazard)
    short_children = expected_rework([3600.0] * 8, hazard)
    assert long_child == pytest.approx(short_children * 8)
    assert expected_rework([], hazard) == 0.0


def test_interruptions_per_second_recovers_the_hazard_behind_a_measured_share():
    durations = [3600.0] * 10
    hazard = interruptions_per_second(0.04, durations)
    # Ten one-hour children at this hazard should lose 4% of one child's worth
    # of exposure, which is where the measured share came from.
    assert hazard * 3600.0 == pytest.approx(0.04)
    assert interruptions_per_second(0.04, []) == 0.0


def test_utilisation_is_the_share_of_rented_vcpus_the_jobs_used(tmp_path):
    path = tmp_path / "instances.tsv"
    # One 8-vCPU instance up for an hour supplies 8 vCPU-hours.
    write_snapshots(
        path,
        [
            (
                "2026-09-19T22:00:00Z",
                "i-1",
                "m5.2xlarge",
                "us-west-2a",
                "running",
                "2026-09-19T21:00:00+00:00",
                "",
            ),
        ],
    )
    fleet = read_fleet_usage(path)
    # Four 2-vCPU jobs running that whole hour would use all of it.
    assert utilisation(job_hours=4.0, fleet=fleet) == pytest.approx(1.0)
    assert utilisation(job_hours=2.0, fleet=fleet) == pytest.approx(0.5)


def test_format_dollars_scales_its_precision():
    assert format_dollars(546.2) == "$546"
    assert format_dollars(5.876) == "$5.88"
    assert format_dollars(0.00108) == "$0.00108"


def test_summarise_sizes_describes_a_set_of_items():
    text = summarise_sizes([1, 2, 4, 8, 16])
    assert "5 items" in text and "31 sheets" in text and "median 4" in text
    assert summarise_sizes([]) == "no items"


def test_cost_model_projection_matches_a_hand_computation():
    model = CostModel(
        fixed_seconds_per_item=189.0,
        seconds_per_page=39.1,
        dollars_per_job_hour=0.0822,
        samples=190,
    )
    projected = model.project(items=35159, pages=441179)
    assert projected["job_hours"] == pytest.approx(
        (35159 * 189.0 + 441179 * 39.1) / 3600
    )
    assert projected["vcpu_hours"] == pytest.approx(projected["job_hours"] * 2)
    assert math.isclose(projected["dollars"], projected["job_hours"] * 0.0822)
