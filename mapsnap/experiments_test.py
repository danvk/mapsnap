"""Tests for mapsnap.experiments."""

import json
from pathlib import Path

from mapsnap.experiments import (
    archive_run,
    auto_run_id,
    build_manifest,
    coverage_counts,
    combined_sha256,
    compute_config_hash,
    diff_volume,
    file_sha256,
    find_dedup_streets_dir,
    format_volume_diff,
    gather_inputs,
    normalize_flags_for_hash,
    page_status,
)


def _write(path: Path, text: str) -> Path:
    """Write text to path, creating parents, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- hashing ---


def test_file_sha256_prefixed_and_stable(tmp_path):
    path = _write(tmp_path / "a.txt", "hello")
    digest = file_sha256(path)
    assert digest.startswith("sha256:")
    assert digest == file_sha256(path)


def test_combined_sha256_order_independent(tmp_path):
    a = _write(tmp_path / "p1.streets.json", "aaa")
    b = _write(tmp_path / "p2.streets.json", "bbb")
    assert combined_sha256([a, b]) == combined_sha256([b, a])


def test_combined_sha256_changes_with_content(tmp_path):
    a = _write(tmp_path / "p1.streets.json", "aaa")
    b = _write(tmp_path / "p2.streets.json", "bbb")
    before = combined_sha256([a, b])
    b.write_text("CHANGED")
    assert combined_sha256([a, b]) != before


# --- flag normalization + config hash ---


def test_normalize_flags_drops_num_workers():
    flags = ["--disable-scale-outlier-check", "--num-workers", "2", "--debug"]
    assert normalize_flags_for_hash(flags) == [
        "--disable-scale-outlier-check",
        "--debug",
    ]


def test_config_hash_ignores_num_workers():
    inputs = {"streets": {"combined_sha": "sha256:x"}}
    a = compute_config_hash(["--num-workers", "1"], inputs)
    b = compute_config_hash(["--num-workers", "8"], inputs)
    assert a == b


def test_config_hash_changes_with_flags():
    inputs = {"streets": {"combined_sha": "sha256:x"}}
    a = compute_config_hash([], inputs)
    b = compute_config_hash(["--disable-scale-outlier-check"], inputs)
    assert a != b
    assert len(a) == 8


def test_config_hash_changes_with_inputs():
    a = compute_config_hash([], {"streets": {"combined_sha": "sha256:x"}})
    b = compute_config_hash([], {"streets": {"combined_sha": "sha256:y"}})
    assert a != b


def test_auto_run_id_truncates_sha():
    assert auto_run_id("1a2b3c4d5e6f", "9f8e7d6c") == "1a2b3c4d-9f8e7d6c"


# --- input gathering + coverage ---


def test_gather_inputs_records_streets_and_truth(tmp_path):
    _write(tmp_path / "p1.streets.json", "s1")
    _write(tmp_path / "p2.streets.json", "s2")
    _write(tmp_path / "p1.boxes.json", "b1")
    centerlines = _write(tmp_path / "centerlines.geojson", "lines")
    truth = _write(tmp_path / "main.iiif.json", "truth")

    inputs = gather_inputs(tmp_path, centerlines, truth)
    assert inputs["streets"]["count"] == 2
    assert inputs["streets"]["combined_sha"].startswith("sha256:")
    assert inputs["truth"] == "main.iiif.json"
    assert inputs["boxes_sha"].startswith("sha256:")


def test_gather_inputs_no_truth(tmp_path):
    _write(tmp_path / "p1.streets.json", "s1")
    centerlines = _write(tmp_path / "centerlines.geojson", "lines")
    inputs = gather_inputs(tmp_path, centerlines, None)
    assert "truth" not in inputs
    assert inputs["boxes_sha"] is None


def test_coverage_counts_classifies_variants(tmp_path):
    for stem in ("p1", "p2", "p3"):
        (tmp_path / f"{stem}.jpg").touch()
    _write(tmp_path / "p1.georef.json", "{}")
    _write(tmp_path / "p2.georef-misscale.json", "{}")
    _write(tmp_path / "p3.georef-1gcp.json", "{}")

    counts = coverage_counts(tmp_path)
    assert counts["pages"] == 3
    assert counts["georeferenced"] == 1
    assert counts["misscale"] == 1
    assert counts["deferred_unconfirmed"] == 1
    assert counts["outlier"] == 0


# --- manifest ---


def test_build_manifest_coverage_only_without_truth(tmp_path):
    (tmp_path / "p1.jpg").touch()
    _write(tmp_path / "p1.georef.json", "{}")
    manifest = build_manifest(
        tmp_path,
        run_id="abc-123",
        flag_tokens=["--debug"],
        inputs={"streets": {"combined_sha": "sha256:x"}},
        git={"sha": "deadbeef", "clean": True},
        command=["mapsnap", "fit", "data/vol"],
        truth_path=None,
        iiif_path=None,
        label="my-run",
    )
    assert manifest["run_id"] == "abc-123"
    assert manifest["label"] == "my-run"
    assert manifest["metrics"]["georeferenced"] == 1
    assert "truth" not in manifest["metrics"]


# --- dedup + archiving ---


def test_find_dedup_streets_dir_matches_combined_sha(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(
        artifacts / "old" / "manifest.json",
        json.dumps({"inputs": {"streets": {"combined_sha": "sha256:match"}}}),
    )
    _write(
        artifacts / "other" / "manifest.json",
        json.dumps({"inputs": {"streets": {"combined_sha": "sha256:nope"}}}),
    )
    found = find_dedup_streets_dir(artifacts, "new", "sha256:match")
    assert found == artifacts / "old"


def test_find_dedup_streets_dir_no_match(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert find_dedup_streets_dir(artifacts, "new", "sha256:x") is None


def test_archive_run_copies_then_symlinks_streets(tmp_path):
    # First run: streets are copied.
    _write(tmp_path / "p1.streets.json", "street-content")
    _write(tmp_path / "p1.georef.json", "{}")
    _write(tmp_path / "p1.txt", "log")
    inputs = {
        "streets": {"combined_sha": combined_sha256([tmp_path / "p1.streets.json"])}
    }

    manifest_a = {"run_id": "run-a", "inputs": inputs}
    run_a = archive_run(tmp_path, "run-a", manifest_a, iiif_path=None, compare_txt=None)
    archived_streets = run_a / "p1.streets.json"
    assert archived_streets.is_file() and not archived_streets.is_symlink()
    assert (run_a / "p1.georef.json").exists()
    assert (run_a / "p1.txt").exists()
    assert (run_a / "manifest.json").exists()

    # Second run with identical streets content: streets are symlinked to run-a's copy.
    manifest_b = {"run_id": "run-b", "inputs": inputs}
    run_b = archive_run(tmp_path, "run-b", manifest_b, iiif_path=None, compare_txt=None)
    linked = run_b / "p1.streets.json"
    assert linked.is_symlink()
    assert linked.read_text() == "street-content"
    assert Path(linked.resolve()) == archived_streets.resolve()


# --- diff ---


def _manifest_with_truth(per_page: dict, **coverage) -> dict:
    """Build a minimal manifest carrying per-page truth RMSE for diff tests."""
    rmses = sorted(per_page.values())
    mean = sum(rmses) / len(rmses) if rmses else 0.0
    metrics = {
        "georeferenced": coverage.get("georeferenced", len(per_page)),
        "misscale": coverage.get("misscale", 0),
        "outlier": coverage.get("outlier", 0),
        "deferred_unconfirmed": coverage.get("deferred_unconfirmed", 0),
        "truth": {
            "compared": len(per_page),
            "mean_rmse_ft": round(mean, 1),
            "median_rmse_ft": rmses[len(rmses) // 2] if rmses else 0.0,
            "per_page": per_page,
        },
    }
    return {"metrics": metrics}


def test_diff_volume_reports_regressions_and_improvements():
    man_a = _manifest_with_truth({"p1": 10.0, "p2": 20.0})
    man_b = _manifest_with_truth({"p1": 15.0, "p2": 12.0})
    diff = diff_volume(man_a, man_b)
    # p1 got worse (regression), p2 got better (improvement).
    assert [r[0] for r in diff["regressions"]] == ["p1"]
    assert [r[0] for r in diff["improvements"]] == ["p2"]
    assert diff["regressions"][0][3] == 5.0
    assert diff["improvements"][0][3] == -8.0


def test_diff_volume_reports_newly_placed_and_failed():
    man_a = _manifest_with_truth({"p1": 10.0, "p3": 30.0})
    man_b = _manifest_with_truth({"p1": 10.0, "p2": 20.0})
    diff = diff_volume(man_a, man_b)
    assert diff["newly_placed"] == ["p2"]
    assert diff["newly_failed"] == ["p3"]


def test_diff_volume_coverage_delta():
    man_a = _manifest_with_truth({"p1": 10.0}, georeferenced=5, misscale=2)
    man_b = _manifest_with_truth({"p1": 10.0}, georeferenced=7, misscale=0)
    diff = diff_volume(man_a, man_b)
    assert diff["coverage"]["georeferenced"] == 2
    assert diff["coverage"]["misscale"] == -2


def test_format_volume_diff_smoke():
    man_a = _manifest_with_truth({"p1": 10.0, "p2": 20.0})
    man_b = _manifest_with_truth({"p1": 15.0, "p2": 12.0})
    text = format_volume_diff("vol_x", diff_volume(man_a, man_b))
    assert "vol_x" in text
    assert "regression" in text
    assert "roll-up" in text


def test_page_status_without_truth_is_ok():
    assert page_status({"metrics": {}}, "p1") == "ok"


def test_page_status_missing_when_absent_from_truth():
    man = _manifest_with_truth({"p1": 10.0})
    assert page_status(man, "p1") == "ok"
    assert page_status(man, "p9") == "missing"
