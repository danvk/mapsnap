"""Tests for the CPU chain driver (mapsnap.loc_fit)."""

from pathlib import Path

from mapsnap.loc_craft import Item
from mapsnap.loc_fit import (
    CENTERLINES_NAME,
    DONE_MARKER,
    UPLOAD_EXCLUDES,
    UPLOAD_GLOBS,
    County,
    plan_fit,
    read_counties,
    upload,
)

ALPHA = Item("sanborn1", "alabama", "1900")


def present(pages: int, *, boxes: bool = True, done: bool = False) -> list[str]:
    """An item's S3 keys with `pages` sheets, optionally crafted or finished."""
    keys = [f"p{n}.jpg" for n in range(1, pages + 1)]
    if boxes:
        keys += [f"p{n}.boxes.json" for n in range(1, pages + 1)]
    if done:
        keys.append(DONE_MARKER)
    return keys


def test_plan_fit_runs_an_item_whose_pages_are_all_crafted() -> None:
    work = plan_fit(ALPHA, present(3), County("US01001"))
    assert work.ready and not work.done
    assert work.pages == ["p1.jpg", "p2.jpg", "p3.jpg"]


def test_plan_fit_skips_an_item_that_already_finished() -> None:
    """The annotation page is written last, so it means the whole chain ran."""
    work = plan_fit(ALPHA, present(3, done=True), County("US01001"))
    assert work.done


def test_plan_fit_waits_for_craft_rather_than_failing() -> None:
    """Boxes missing means the GPU pass has not been through yet, not an error."""
    keys = present(3)
    keys.remove("p2.boxes.json")
    work = plan_fit(ALPHA, keys, County("US01001"))
    assert not work.ready and not work.done
    assert "await craft" in work.reason


def test_plan_fit_waits_when_no_county_extract_is_known() -> None:
    work = plan_fit(ALPHA, present(2), None)
    assert not work.ready
    assert "county" in work.reason


def test_plan_fit_reports_an_item_the_mirror_never_produced() -> None:
    work = plan_fit(ALPHA, ["metadata.json"], County("US01001"))
    assert not work.ready
    assert "no pages" in work.reason


def test_plan_fit_ignores_panels_when_listing_parent_pages() -> None:
    """Panels are cut locally each run; the parents are what gets planned."""
    keys = present(2) + ["p1__1.jpg", "p1__2.jpg", "p1.panels.json"]
    work = plan_fit(ALPHA, keys, County("US01001"))
    assert work.pages == ["p1.jpg", "p2.jpg"]


def test_county_key_points_at_the_extract() -> None:
    assert County("US06037").key == "osm-by-county/US06037.osm.pbf"


def test_read_counties_merges_both_mappings(tmp_path: Path) -> None:
    """Counties come from items.tsv, independent cities from city-items.tsv."""
    items = tmp_path / "items.tsv"
    items.write_text(
        "item\tstate\tcounty\tcity\tsheets\tmatch\tfips\tne_name\n"
        "sanborn1\talabama\tlimestone county\tathens\t1\texact\tUS01083\tLimestone\n"
    )
    cities = tmp_path / "city-items.tsv"
    cities.write_text(
        "item\tstate\tcounty\tcity\tsheets\tmatch\tfips\tosm_relation\tosm_name\n"
        "sanborn2\tvirginia\tindependent cities\trichmond\t27\texact\tUS51760\t3864712\tRichmond\n"
    )
    counties = read_counties([items, cities])
    assert counties == {"sanborn1": County("US01083"), "sanborn2": County("US51760")}


def test_read_counties_skips_a_row_with_no_fips(tmp_path: Path) -> None:
    path = tmp_path / "items.tsv"
    path.write_text("item\tfips\nsanborn1\t\nsanborn2\tUS01083\n")
    assert read_counties([path]) == {"sanborn2": County("US01083")}


def test_upload_excludes_the_panel_files_and_the_extract(monkeypatch) -> None:
    """Panel images are regenerable; the extract is 20 MB of someone else's data."""
    from mapsnap import loc_fit

    calls: list[list[str]] = []
    monkeypatch.setattr(loc_fit, "run_aws", lambda command, **kw: calls.append(command))
    upload(Path("/tmp/x"), "s3://bucket", ALPHA)
    command = calls[0]
    assert command[:3] == ["aws", "s3", "sync"]
    excluded = {command[i + 1] for i, word in enumerate(command) if word == "--exclude"}
    assert CENTERLINES_NAME in excluded
    for pattern in UPLOAD_EXCLUDES:
        assert pattern in excluded


def test_upload_patterns_do_not_catch_what_must_be_kept() -> None:
    """A too-greedy exclude would silently drop the reads or the cut."""
    import fnmatch

    keepers = [
        "p209.panels.json",
        "p209__1.streets.json",
        "p209__1.georef-final.json",
        "p220.streets.json",
        "adjacency.json",
        "raw/p0.keymap.json",
        DONE_MARKER,
    ]
    for name in keepers:
        for pattern in UPLOAD_EXCLUDES:
            assert not fnmatch.fnmatch(name, pattern), f"{pattern} would drop {name}"


def test_upload_patterns_do_catch_the_regenerable_files() -> None:
    for name in ["p209__1.jpg", "p209__12.jpg", "p209__1.boxes.json"]:
        assert any(
            __import__("fnmatch").fnmatch(name, pattern) for pattern in UPLOAD_EXCLUDES
        ), name


def test_upload_globs_and_excludes_do_not_contradict() -> None:
    """The globs document intent; the excludes enforce it. They must agree."""
    import fnmatch

    for glob in UPLOAD_GLOBS:
        for pattern in UPLOAD_EXCLUDES:
            assert not fnmatch.fnmatch(glob, pattern), f"{pattern} excludes {glob}"


def test_run_chain_derives_panel_boxes_and_reads_effective_pages(
    tmp_path: Path, monkeypatch
) -> None:
    """split, then a derive-only craft, then adjacency, keymap, ocr, fit.

    The smoke run of 2026-09-16 failed ocr with "No CRAFT boxes for p10__1.jpg":
    split cuts a panel's image and P(road) crop but not its boxes.
    """
    from mapsnap import loc_fit
    from mapsnap.keymap.records import write_keymaps_record

    for name in ("p1.jpg", "p1__1.jpg", "p1__2.jpg", "p2.jpg"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "raw").mkdir()
    for name in ("p0.jpg", "p0__1.jpg"):
        (tmp_path / "raw" / name).write_bytes(b"")
    write_keymaps_record(tmp_path, ["p0__1"])

    commands: list[list[str]] = []
    monkeypatch.setattr(
        loc_fit, "stage", lambda command, local: commands.append(command)
    )
    work = plan_fit(
        ALPHA, ["p1.jpg", "p2.jpg", "p1.boxes.json", "p2.boxes.json"], County("US01001")
    )
    loc_fit.run_chain(tmp_path, work, "https://images.test/x")

    assert [c[1] for c in commands] == [
        "split",
        "craft",
        "adjacency",
        "keymap",
        "ocr",
        "fit",
    ]
    names = lambda c: {Path(a).name for a in c if a.endswith(".jpg")}
    assert names(commands[0]) == {"p1.jpg", "p2.jpg"}  # split runs on the parents
    craft = commands[1]
    assert "--resume" in craft
    assert names(craft) == {"p1__1.jpg", "p1__2.jpg", "p2.jpg", "p0.jpg", "p0__1.jpg"}
    assert commands[3][2:] == [str(tmp_path / "raw" / "p0__1.jpg")]  # the recorded key
    assert names(commands[4]) == {
        "p1__1.jpg",
        "p1__2.jpg",
        "p2.jpg",
    }  # not the split parent
    assert "--image-base-url" in commands[5]
