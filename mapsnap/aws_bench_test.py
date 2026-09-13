"""Tests for the pure parts of mapsnap.aws_bench (timing math, layout, report)."""

from pathlib import Path

from mapsnap.aws_bench import (
    Bench,
    clear_sidecars,
    format_report,
    machine_label,
    parent_pages,
    per_page_seconds,
    raw_sheet,
    stage_scratch,
    startup_seconds,
)


def test_per_page_subtracts_the_one_page_run() -> None:
    assert per_page_seconds(6.0, 26.0, 11) == 2.0
    assert per_page_seconds(6.0, 6.0, 1) == 6.0


def test_startup_is_the_one_page_run_minus_its_page() -> None:
    assert startup_seconds(6.0, 2.0) == 4.0
    assert startup_seconds(1.0, 2.0) == 0.0


def make_volume(tmp_path: Path) -> Path:
    volume = tmp_path / "vol"
    (volume / "raw").mkdir(parents=True)
    for name in ("p10", "p2", "p2__1", "p2__2", "p1A"):
        (volume / f"{name}.jpg").write_bytes(b"jpg")
    (volume / "centerlines.geojson").write_text("{}")
    (volume / "raw" / "p0.jpg").write_bytes(b"raw")
    return volume


def test_parent_pages_skips_panels_and_sorts_by_page_key(tmp_path: Path) -> None:
    volume = make_volume(tmp_path)
    assert [path.stem for path in parent_pages(volume)] == ["p1A", "p2", "p10"]
    assert raw_sheet(volume) == volume / "raw" / "p0.jpg"
    assert raw_sheet(tmp_path) is None


def test_stage_scratch_symlinks_inputs_only(tmp_path: Path) -> None:
    volume = make_volume(tmp_path)
    scratch = stage_scratch(volume, tmp_path / "work")
    assert scratch == tmp_path / "work" / "vol"
    assert sorted(path.name for path in scratch.iterdir()) == [
        "centerlines.geojson",
        "p10.jpg",
        "p1A.jpg",
        "p2.jpg",
        "raw",
    ]
    assert (scratch / "p2.jpg").is_symlink()
    assert (scratch / "raw" / "p0.jpg").read_bytes() == b"raw"
    stage_scratch(volume, tmp_path / "work")  # idempotent


def test_clear_sidecars_removes_only_named_suffixes(tmp_path: Path) -> None:
    page = tmp_path / "p3.jpg"
    for suffix in ("jpg", "boxes.json", "streets.json", "txt"):
        (tmp_path / f"p3.{suffix}").write_text("")
    clear_sidecars([page], ("streets.json", "txt"))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "p3.boxes.json",
        "p3.jpg",
    ]


def test_report_lines_up_machines_and_fills_gaps() -> None:
    laptop = {
        "machine": {"hostname": "mbp", "accelerator": "mps"},
        "stages": [
            {"stage": "craft", "device": "mps", "workers": 1, "per_page": 4.5},
            {"stage": "craft", "device": "cpu", "workers": 1, "per_page": 25.0},
        ],
    }
    cloud = {
        "machine": {
            "instance_type": "g4dn.xlarge",
            "gpu": "Tesla T4",
            "accelerator": "cuda",
        },
        "stages": [
            {"stage": "craft", "device": "cuda", "workers": 1, "per_page": 1.5},
            {"stage": "craft", "device": "cpu", "workers": 1, "per_page": 40.0},
            {"stage": "ocr", "device": "cuda", "workers": 4, "per_page": 0.9},
            {"stage": "ocr", "device": "cpu", "workers": 1, "per_page": None},
        ],
    }
    assert machine_label(laptop["machine"]) == "mbp (mps)"
    assert machine_label(cloud["machine"]) == "g4dn.xlarge (Tesla T4)"
    report = format_report([laptop, cloud])
    lines = report.splitlines()
    assert lines[0].split() == [
        "stage",
        "device",
        "wkr",
        "mbp",
        "(mps)",
        "g4dn.xlarge",
        "(Tesla",
        "T4)",
    ]
    assert lines[1].split() == ["craft", "cpu", "1", "25.00", "40.00"]
    assert lines[2].split() == ["craft", "cuda", "1", "-", "1.50"]
    assert lines[3].split() == ["craft", "mps", "1", "4.50", "-"]
    assert lines[4].split() == ["ocr", "cpu", "1", "-", "fail"]
    assert lines[5].split() == ["ocr", "cuda", "4", "-", "0.90"]


def test_a_failing_stage_is_recorded_and_the_bench_continues(tmp_path: Path) -> None:
    page = tmp_path / "p1.jpg"
    page.write_bytes(b"jpg")
    bench = Bench(tmp_path, tmp_path / "bench.log", workers=1)
    bench.cli_stage("craft", "cpu", [page], lambda pages: ["false"], ())
    bench.single_run("ocr", "cpu", [page], ["true"], ())
    assert [(r.stage, r.per_page is None) for r in bench.results] == [
        ("craft", True),
        ("ocr", False),
    ]
    assert bench.results[0].note.startswith("CalledProcessError")
