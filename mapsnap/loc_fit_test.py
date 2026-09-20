"""Tests for the CPU chain driver (mapsnap.loc_fit)."""

from pathlib import Path

from mapsnap.loc_craft import Item
from mapsnap.loc_fit import (
    ARCHIVE_TAG,
    CENTERLINES_NAME,
    DONE_MARKER,
    RUNS_DIRNAME,
    UPLOAD_EXCLUDES,
    UPLOAD_GLOBS,
    UPLOAD_INCLUDES,
    County,
    plan_fit,
    read_counties,
    upload,
)

ALPHA = Item("sanborn1", "alabama", "1900")
TAG = "v1.3"


def present(pages: int, *, boxes: bool = True, done: bool = False) -> list[str]:
    """An item's S3 keys with `pages` sheets, optionally crafted or finished."""
    keys = [f"p{n}.jpg" for n in range(1, pages + 1)]
    if boxes:
        keys += [f"p{n}.boxes.json" for n in range(1, pages + 1)]
    if done:
        keys.append(f"{RUNS_DIRNAME}/{TAG}/{DONE_MARKER}")
    return keys


def test_plan_fit_runs_an_item_whose_pages_are_all_crafted() -> None:
    work = plan_fit(ALPHA, present(3), County("US01001"), TAG)
    assert work.ready and not work.done
    assert work.pages == ["p1.jpg", "p2.jpg", "p3.jpg"]


def test_plan_fit_skips_an_item_that_already_finished() -> None:
    """The annotation page is written last, so it means the whole chain ran."""
    work = plan_fit(ALPHA, present(3, done=True), County("US01001"), TAG)
    assert work.done


def test_plan_fit_waits_for_craft_rather_than_failing() -> None:
    """Boxes missing means the GPU pass has not been through yet, not an error."""
    keys = present(3)
    keys.remove("p2.boxes.json")
    work = plan_fit(ALPHA, keys, County("US01001"), TAG)
    assert not work.ready and not work.done
    assert "await craft" in work.reason


def test_plan_fit_waits_when_no_county_extract_is_known() -> None:
    work = plan_fit(ALPHA, present(2), None, TAG)
    assert not work.ready
    assert "county" in work.reason


def test_plan_fit_reports_an_item_the_mirror_never_produced() -> None:
    """And settles it: no worker will ever find pages the mirror does not hold,
    so putting it back only buys the next one the same dead end. The 45 items
    whose sheets are all non-numeric (Des Moines 1906's lone pcbd sheet among
    them) rode the test-200b queue into its dead-letter queue this way."""
    work = plan_fit(ALPHA, ["metadata.json"], County("US01001"), TAG)
    assert not work.ready
    assert "no pages" in work.reason
    assert work.unprocessable


def test_plan_fit_keeps_the_fixable_not_ready_cases_retryable() -> None:
    # Someone else clears these -- the GPU pass, or an uploaded extract -- so
    # they go back on the queue and a later worker gets them.
    waiting_on_craft = present(3)
    waiting_on_craft.remove("p2.boxes.json")
    for keys, county in ((waiting_on_craft, County("US01001")), (present(2), None)):
        work = plan_fit(ALPHA, keys, county, TAG)
        assert not work.ready
        assert not work.unprocessable


def test_plan_fit_ignores_panels_when_listing_parent_pages() -> None:
    """Panels are cut locally each run; the parents are what gets planned."""
    keys = present(2) + ["p1__1.jpg", "p1__2.jpg", "p1.panels.json"]
    work = plan_fit(ALPHA, keys, County("US01001"), TAG)
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
    upload(Path("/tmp/x"), "s3://bucket", ALPHA, TAG)
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
    import fnmatch

    for name in ["p209__1.jpg", "p209__12.jpg", "p209__1.boxes.json"]:
        assert any(fnmatch.fnmatch(name, pattern) for pattern in UPLOAD_EXCLUDES), name


def test_upload_keeps_the_candidate_files() -> None:
    """They record what snap and street-solve rejected; re-running the search to
    recover that is the expensive part, and Madison p20__3 needed it."""
    import fnmatch

    for name in (
        "artifacts/osm_snap/candidates.jsonl",
        "artifacts/street_solve/candidates.jsonl",
    ):
        assert name in UPLOAD_GLOBS
        for pattern in UPLOAD_EXCLUDES:
            assert not fnmatch.fnmatch(name, pattern), f"{pattern} would drop {name}"


def test_upload_still_drops_the_reconcile_report() -> None:
    """verdicts.jsonl and report.md restate what the provenance records carry."""
    import fnmatch

    for name in ("artifacts/reconcile/verdicts.jsonl", "artifacts/reconcile/report.md"):
        assert any(fnmatch.fnmatch(name, pattern) for pattern in UPLOAD_EXCLUDES), name


def test_upload_globs_and_excludes_do_not_contradict() -> None:
    """The globs document intent; the filters enforce it. They must agree.

    An exclude may still cover a glob as long as an include wins it back --
    that is how the run manifest survives the exclusion of the archive it
    sits in.
    """
    import fnmatch

    for glob in UPLOAD_GLOBS:
        excluded = any(fnmatch.fnmatch(glob, p) for p in UPLOAD_EXCLUDES)
        rescued = any(fnmatch.fnmatch(glob, p) for p in UPLOAD_INCLUDES)
        assert not excluded or rescued, f"{glob} is excluded and never included"


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
    commands: list[list[str]] = []

    def record(command, local, **kw):
        commands.append(command)
        # The chain identifies the key map itself now, so the record appears
        # when keymap-detect runs rather than being seeded beforehand.
        if command[1] == "keymap-detect":
            write_keymaps_record(tmp_path, ["p0__1"])

    monkeypatch.setattr(loc_fit, "stage", record)
    work = plan_fit(
        ALPHA,
        ["p1.jpg", "p2.jpg", "p1.boxes.json", "p2.boxes.json"],
        County("US01001"),
        TAG,
    )
    loc_fit.run_chain(tmp_path, work, work.run_tag)

    assert [c[1] for c in commands] == [
        "split",
        "keymap-detect",
        "craft",
        "adjacency",
        "keymap",
        "ocr",
        "fit",
    ]
    names = lambda c: {Path(a).name for a in c if a.endswith(".jpg")}
    assert names(commands[0]) == {"p1.jpg", "p2.jpg"}  # split runs on the parents
    by_stage = {c[1]: c for c in commands}
    craft = by_stage["craft"]
    assert "--resume" in craft
    assert names(craft) == {"p1__1.jpg", "p1__2.jpg", "p2.jpg", "p0.jpg", "p0__1.jpg"}
    # the recorded key, now identified by this run rather than read from S3
    assert by_stage["keymap"][2:] == [str(tmp_path / "raw" / "p0__1.jpg")]
    assert names(by_stage["ocr"]) == {
        "p1__1.jpg",
        "p1__2.jpg",
        "p2.jpg",
    }  # not the split parent
    assert by_stage["fit"][:5] == ["mapsnap", "fit", str(tmp_path), "--tag", "mapsnap"]


def test_resolve_counties_downloads_s3_urls_by_basename(
    tmp_path: Path, monkeypatch
) -> None:
    """Two mapping files must not collide on a fixed download name."""
    from mapsnap import loc_fit
    from mapsnap.loc_fit import resolve_counties

    calls: list[list[str]] = []

    def fake_aws(command, **kwargs):
        calls.append(command)
        Path(command[4]).write_text("item\tfips\n")

    monkeypatch.setattr(loc_fit, "run_aws", fake_aws)
    work = tmp_path / "work"
    paths = resolve_counties(
        ["s3://b/_craft/items.tsv", "s3://b/_craft/city-items.tsv", "/local/x.tsv"],
        work,
    )
    assert paths == [work / "items.tsv", work / "city-items.tsv", Path("/local/x.tsv")]
    assert len(calls) == 2
    resolve_counties(["s3://b/_craft/items.tsv"], work)
    assert len(calls) == 2  # already downloaded


def test_parser_accepts_gpu_as_a_no_op() -> None:
    """bootstrap.sh passes --gpu to every job on a GPU box; the chain must not choke."""
    from mapsnap.loc_fit import build_parser

    args = build_parser().parse_args(
        ["--counties", "a.tsv", "b.tsv", "--gpu", "--queue", "https://q"]
    )
    assert args.gpu is True
    assert args.counties == ["a.tsv", "b.tsv"]


def test_upload_keeps_the_run_manifest_but_not_the_rest_of_the_archive() -> None:
    """fit archives a second copy of every sidecar; only its manifest is worth it."""
    import fnmatch

    manifest = "artifacts/mapsnap/manifest.json"
    duplicate = "artifacts/mapsnap/p1.streets.json"
    assert any(fnmatch.fnmatch(manifest, p) for p in UPLOAD_EXCLUDES)
    assert manifest in UPLOAD_INCLUDES  # the include is applied last and wins
    assert any(fnmatch.fnmatch(duplicate, p) for p in UPLOAD_EXCLUDES)
    assert not any(fnmatch.fnmatch(duplicate, p) for p in UPLOAD_INCLUDES)


def test_upload_orders_includes_after_excludes(monkeypatch) -> None:
    """aws s3 sync takes the last matching filter, so order is the behaviour."""
    from mapsnap import loc_fit

    calls: list[list[str]] = []
    monkeypatch.setattr(loc_fit, "run_aws", lambda command, **kw: calls.append(command))
    upload(Path("/tmp/x"), "s3://bucket", ALPHA, TAG)
    command = calls[0]
    last_exclude = max(i for i, w in enumerate(command) if w == "--exclude")
    first_include = min(i for i, w in enumerate(command) if w == "--include")
    assert first_include > last_exclude


def test_run_chain_clears_the_previous_archive(tmp_path, monkeypatch) -> None:
    """fit refuses to overwrite an archive, and the sync brings the old one down."""
    from mapsnap import loc_fit

    (tmp_path / "p1.jpg").write_bytes(b"")
    stale = tmp_path / "artifacts" / "mapsnap"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}")
    monkeypatch.setattr(loc_fit, "stage", lambda command, local, **kw: None)
    work = plan_fit(ALPHA, ["p1.jpg", "p1.boxes.json"], County("US01001"), TAG)
    loc_fit.run_chain(tmp_path, work)
    assert not stale.exists()


def test_plan_fit_is_done_only_for_its_own_run() -> None:
    """Another run's marker says nothing about this one -- the point of tags."""
    keys = present(3, done=True)
    assert plan_fit(ALPHA, keys, County("US01001"), TAG).done
    assert not plan_fit(ALPHA, keys, County("US01001"), "other").done


def test_plan_fit_ignores_a_bare_marker_at_the_item_root() -> None:
    """Pre-tag layout: a top-level marker must not retire a tagged run."""
    keys = present(3) + [DONE_MARKER]
    assert not plan_fit(ALPHA, keys, County("US01001"), TAG).done


def test_upload_writes_the_done_marker_after_everything_else(monkeypatch, tmp_path):
    """The marker is what retires an item, so it must not precede what it vouches for.

    In one sync it sorts before `p*.streets.json` and landed first: an upload
    interrupted in between left a done marker over a partial item.
    """
    from mapsnap import loc_fit

    (tmp_path / DONE_MARKER).write_text("{}")
    calls: list[list[str]] = []
    monkeypatch.setattr(loc_fit, "run_aws", lambda command, **kw: calls.append(command))
    loc_fit.upload(tmp_path, "s3://bucket", ALPHA, TAG)

    assert [c[2] for c in calls] == ["sync", "cp"]
    assert "--exclude" in calls[0] and DONE_MARKER in calls[0]
    assert calls[1][-2].endswith(f"{RUNS_DIRNAME}/{TAG}/{DONE_MARKER}")


def test_upload_targets_the_run_directory_and_skips_the_stable_half(
    monkeypatch, tmp_path
):
    """Images and CRAFT boxes are written once at the item root and shared."""
    from mapsnap import loc_fit

    calls: list[list[str]] = []
    monkeypatch.setattr(loc_fit, "run_aws", lambda command, **kw: calls.append(command))
    loc_fit.upload(tmp_path, "s3://bucket", ALPHA, TAG)

    assert calls[0][4].endswith(f"/{RUNS_DIRNAME}/{TAG}")
    for pattern in ("*.jpg", "*.boxes.json", "metadata.json"):
        assert pattern in calls[0], pattern
    # The fit archive's manifest is re-included, and must stay last to win.
    assert calls[0].index("--include") > calls[0].index(f"artifacts/{ARCHIVE_TAG}/*")


def test_resolve_run_tag_takes_the_message_as_the_authority() -> None:
    """One value, so the S3 prefix and the recorded provenance cannot disagree."""
    from mapsnap.loc_fit import resolve_run_tag

    assert resolve_run_tag("v1.3", None) == "v1.3"
    assert resolve_run_tag("v1.3", "v1.3") == "v1.3"
    assert resolve_run_tag(None, "v1.3") == "v1.3"


def test_resolve_run_tag_stops_a_worker_pointed_at_the_wrong_queue() -> None:
    """A mismatch is a launch error, not something to paper over."""
    import pytest

    from mapsnap.loc_fit import resolve_run_tag

    with pytest.raises(ValueError, match="wrong queue|queue says"):
        resolve_run_tag("v1.3", "v1.2")
    with pytest.raises(ValueError, match="no run tag"):
        resolve_run_tag(None, None)


def test_fetch_item_does_not_pull_other_runs(monkeypatch, tmp_path):
    """A recursive sync of the item prefix would drag every run down with it."""
    from mapsnap import loc_fit

    syncs: list[tuple] = []
    monkeypatch.setattr(loc_fit, "sync", lambda *a: syncs.append(a))
    monkeypatch.setattr(loc_fit, "run_aws", lambda command, **kw: None)
    work = plan_fit(ALPHA, present(2), County("US01001"), TAG)
    loc_fit.fetch_item(work, "s3://bucket", tmp_path)

    stable = syncs[0]
    assert stable[2:] == ("--exclude", f"{RUNS_DIRNAME}/*")
    # This run's own outputs land last, so an interrupted item resumes.
    assert syncs[-1][0].endswith(f"{RUNS_DIRNAME}/{TAG}")


def test_borrow_reads_refuses_a_run_that_read_different_streets(monkeypatch, tmp_path):
    """The county re-cut changed every extract; reads made against the old one
    match streets that are no longer in the file."""
    import json as _json

    from mapsnap import loc_fit

    (tmp_path / CENTERLINES_NAME).write_bytes(b"new extract")
    monkeypatch.setattr(
        loc_fit,
        "run_aws",
        lambda command, **kw: type(
            "R",
            (),
            {"stdout": _json.dumps({"inputs": {"centerlines_sha": "sha256:old"}})},
        )(),
    )
    syncs: list[tuple] = []
    monkeypatch.setattr(loc_fit, "sync", lambda *a: syncs.append(a))
    loc_fit.borrow_reads(tmp_path, "s3://bucket", ALPHA, "v1.2")
    assert syncs == []


def test_borrow_reads_takes_only_the_reads(monkeypatch, tmp_path):
    """Poses and provenance are this run's to make, even when the reads are lent."""
    import json as _json

    from mapsnap import experiments, loc_fit

    (tmp_path / CENTERLINES_NAME).write_bytes(b"same extract")
    sha = experiments.file_sha256(tmp_path / CENTERLINES_NAME)
    monkeypatch.setattr(
        loc_fit,
        "run_aws",
        lambda command, **kw: type(
            "R", (), {"stdout": _json.dumps({"inputs": {"centerlines_sha": sha}})}
        )(),
    )
    syncs: list[tuple] = []
    monkeypatch.setattr(loc_fit, "sync", lambda *a: syncs.append(a))
    loc_fit.borrow_reads(tmp_path, "s3://bucket", ALPHA, "v1.2")

    assert len(syncs) == 1
    filters = syncs[0][2:]
    assert filters[:2] == ("--exclude", "*")
    assert "p*.streets.json" in filters and "p*.georef.json" not in filters


def test_one_item_touches_s3_in_the_right_order(monkeypatch, tmp_path):
    """The whole lifecycle, as S3 sees it: read the stable half, borrow reads,
    resume this run, then write the outputs and only then the done marker."""
    import json as _json

    from mapsnap import experiments, loc_fit

    local = tmp_path / ALPHA.item
    events: list[str] = []
    extract_sha = experiments.file_sha256

    def tail(url: str) -> str:
        return url.split(ALPHA.item)[-1] or "/"

    def fake_sync(source, destination, *filters):
        joined = " ".join(filters)
        events.append(f"GET {tail(source)} {joined}".rstrip())

    def fake_run_aws(command, **kw):
        if command[2] == "cp" and command[3].endswith("manifest.json"):
            sha = extract_sha(local / CENTERLINES_NAME)
            return type(
                "R", (), {"stdout": _json.dumps({"inputs": {"centerlines_sha": sha}})}
            )()
        if command[2] == "cp" and command[4].endswith(CENTERLINES_NAME):
            Path(command[4]).write_bytes(b"extract")  # the county extract landing
        elif command[2] == "cp":
            events.append(f"CP {tail(command[4])}")
        elif command[2] == "sync":
            events.append(f"PUT {tail(command[4])}")
        return type("R", (), {"stdout": ""})()

    monkeypatch.setattr(loc_fit, "sync", fake_sync)
    monkeypatch.setattr(loc_fit, "run_aws", fake_run_aws)
    monkeypatch.setattr(loc_fit, "run_chain", lambda *a: None)

    work = plan_fit(ALPHA, present(2), County("US01001"), TAG)
    loc_fit.fetch_item(work, "s3://bucket", tmp_path, ocr_from="v1.2")
    (local / DONE_MARKER).write_text("{}")
    loc_fit.process_item(work, local, "s3://bucket")

    reads = "--exclude * --include p*.streets.json --include p*.txt"
    assert events == [
        f"GET / --exclude {RUNS_DIRNAME}/*",
        f"GET /{RUNS_DIRNAME}/v1.2 {reads}",
        f"GET /{RUNS_DIRNAME}/{TAG}",
        f"PUT /{RUNS_DIRNAME}/{TAG}",
        f"CP /{RUNS_DIRNAME}/{TAG}/{DONE_MARKER}",
    ]


def test_check_args_validates_without_touching_anything(capsys, monkeypatch) -> None:
    """A worker's flags must be checkable before a fleet boots on them.

    The first test-200 launch died because `--counties` is required and the
    launcher did not pass it: three instances booted, argparse refused, and the
    queue was never touched. Nothing here may reach AWS.
    """
    import sys

    from mapsnap import loc_fit

    def explode(*args, **kwargs):
        raise AssertionError("--check-args must not touch the network")

    monkeypatch.setattr(loc_fit, "run_aws", explode)
    monkeypatch.setattr(loc_fit, "resolve_manifest", explode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mapsnap loc-fit",
            "--check-args",
            "--queue",
            "https://sqs.example/q",
            "--run-tag",
            "v1.3",
            "--counties",
            "s3://b/_craft/items.tsv",
            "s3://b/_craft/city-items.tsv",
        ],
    )
    loc_fit.main()
    assert "arguments OK" in capsys.readouterr().out


def test_check_args_still_requires_the_counties(monkeypatch) -> None:
    """The flag whose absence killed the first launch is the one to catch."""
    import sys

    import pytest

    from mapsnap import loc_fit

    monkeypatch.setattr(
        sys, "argv", ["mapsnap loc-fit", "--check-args", "--queue", "https://q"]
    )
    with pytest.raises(SystemExit) as caught:
        loc_fit.main()
    assert caught.value.code != 0


def test_upload_keeps_the_keymap_annotation_page() -> None:
    """The key map is georeferenced like any other sheet and is worth publishing."""
    import fnmatch

    name = f"{ARCHIVE_TAG}.keymap.iiif.json"
    assert name in UPLOAD_GLOBS
    for pattern in UPLOAD_EXCLUDES:
        assert not fnmatch.fnmatch(name, pattern), f"{pattern} would drop {name}"


def test_the_done_marker_is_not_the_keymap_page() -> None:
    """Both end in .iiif.json; retiring an item on the wrong one would be silent."""
    assert DONE_MARKER == f"{ARCHIVE_TAG}.iiif.json"
    assert DONE_MARKER != f"{ARCHIVE_TAG}.keymap.iiif.json"


def test_keymap_is_identified_after_the_split(monkeypatch, tmp_path):
    """The mirror's keymaps.json names whole sheets; a split volume needs panels.

    `loc-keymaps` runs before anything is split, so it named Los Angeles 1949
    vol 14's key map `p0a` -- the whole sheet, key map AND p1499 inset. A local
    run splits first and names `pa__2`, the panel that is actually the key map.
    Identifying here, after the split, is what makes the two agree.
    """
    from mapsnap import loc_fit

    commands: list[list[str]] = []
    monkeypatch.setattr(
        loc_fit, "stage", lambda command, local, **kw: commands.append(command)
    )
    work = plan_fit(ALPHA, present(2), County("US01001"), TAG)
    loc_fit.run_chain(tmp_path, work, TAG)

    names = [c[1] for c in commands]
    assert names.index("keymap-detect") > names.index("split"), (
        "identification must see the panels the split produced"
    )
    assert names.index("keymap-detect") < names.index("craft"), (
        "craft derives boxes for the raw key-map sheets this names"
    )
    assert "keymap-detect" in names and names.index("keymap-detect") < (
        names.index("ocr")
    )


def test_keymap_detect_is_given_the_volume_not_the_pages(monkeypatch, tmp_path):
    """It takes a volume directory and writes keymaps.json into it."""
    from mapsnap import loc_fit

    commands: list[list[str]] = []
    monkeypatch.setattr(
        loc_fit, "stage", lambda command, local, **kw: commands.append(command)
    )
    work = plan_fit(ALPHA, present(2), County("US01001"), TAG)
    loc_fit.run_chain(tmp_path, work, TAG)

    detect = next(c for c in commands if c[1] == "keymap-detect")
    assert detect[2:] == [str(tmp_path)]


def test_the_mirrors_keymap_record_is_dropped_before_identifying(monkeypatch, tmp_path):
    """A stale whole-sheet record must not survive a failed identification."""
    from mapsnap import loc_fit

    stale = tmp_path / "keymaps.json"
    stale.write_text('{"keys": ["p0a"]}')
    seen: list[bool] = []
    monkeypatch.setattr(
        loc_fit,
        "stage",
        lambda command, local, **kw: (
            seen.append(stale.exists()) if command[1] == "keymap-detect" else None
        ),
    )
    work = plan_fit(ALPHA, present(2), County("US01001"), TAG)
    loc_fit.run_chain(tmp_path, work, TAG)
    assert seen == [False], "the mirror's record is gone before identification runs"


def test_no_key_map_is_an_answer_not_a_failure(monkeypatch, tmp_path):
    """keymap-detect exits 1 when it finds none, which most volumes are.

    Failing the item on that broke the test-200b run: volumes with no key map
    failed outright, returned to the queue, and failed again.
    """
    import subprocess

    from mapsnap import loc_fit

    monkeypatch.setattr(
        loc_fit.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 1, "", "No key map identified."
        ),
    )
    (tmp_path / "keymaps.json").write_text('{"keys": []}')
    loc_fit.stage(
        ["mapsnap", "keymap-detect", str(tmp_path)],
        tmp_path,
        ok_if=lambda: (tmp_path / "keymaps.json").exists(),
    )  # must not raise


def test_a_crash_with_no_record_is_still_a_failure(monkeypatch, tmp_path):
    """The record's presence is the signal, not the exit code."""
    import subprocess

    import pytest

    from mapsnap import loc_fit

    monkeypatch.setattr(
        loc_fit.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 1, "", "Traceback: model not found"
        ),
    )
    with pytest.raises(OSError, match="keymap-detect failed"):
        loc_fit.stage(
            ["mapsnap", "keymap-detect", str(tmp_path)],
            tmp_path,
            ok_if=lambda: (tmp_path / "keymaps.json").exists(),
        )


def test_prepare_next_settles_an_unprocessable_item_and_releases_the_rest(
    monkeypatch, tmp_path: Path
) -> None:
    """The half that matters: which queue callback each not-ready item gets.
    Releasing one the mirror will never hold is what dead-lettered Des Moines
    1906 -- ten receives, then the DLQ -- while a genuinely transient item must
    still go back so a later worker can take it."""
    from mapsnap import loc_fit

    missing = Item("sanborn02629_005", "iowa", "1906")
    uncrafted = Item("sanborn2", "alabama", "1900")
    listings = {
        missing.prefix: ["metadata.json"],
        uncrafted.prefix: present(1, boxes=False),
    }
    monkeypatch.setattr(loc_fit, "list_prefix", lambda bucket, prefix: listings[prefix])

    retired: list[str] = []
    released: list[str] = []
    prepared = loc_fit.prepare_next(
        iter(list(enumerate([missing, uncrafted], start=1))),
        "s3://bucket",
        tmp_path,
        counties={item.item: County("US01001") for item in (missing, uncrafted)},
        tag_for=lambda item: TAG,
        fetch=False,
        retire=lambda item: retired.append(item.item),
        release=lambda item: released.append(item.item),
    )
    assert retired == [missing.item], "settled, not handed to the next worker"
    assert released == [uncrafted.item], "the GPU pass will catch up"
    assert prepared.unprocessable == 1 and prepared.waiting == 1
    assert prepared.work is None


def test_failure_tail_keeps_the_error_not_the_progress_bar() -> None:
    """Miami's key-map failure (sanborn01309_018) was reported to the corpus log
    as three lines of detector thresholds, because a tqdm bar writes to stderr
    and the reporter kept the last six lines. The answer -- "Could not derive a
    --pages spec" -- sat just above the cut and was thrown away, so the item
    could not be diagnosed from the run at all."""
    from mapsnap.loc_fit import failure_tail

    output = (
        "Using centerlines: /opt/craft/scratch-0/centerlines.geojson\n"
        "Could not derive a --pages spec from the volume's page images; pass --pages.\n"
        "  0%|          | 0/1 [00:00<?, ?it/s]"
        "\rBlock index: 91370 segments across 10521 streets\n"
        "Auto min-short-side: 26.0px (p25 of confidence>=0.5 detections)\n"
        "Thresholds: min_confidence=0.15 min_long_side=52.0px min_short_side=26.0px\n"
        "  0%|          | 0/1 [00:00<?, ?it/s]\n"
        " 50%|#####     | 1/2 [00:01<00:01,  1.2s/it]\n"
    )
    tail = failure_tail(output)
    assert "Could not derive a --pages spec" in tail
    assert "it/s]" not in tail and "%|" not in tail
    # The surviving non-progress lines stay, in order.
    assert tail.index("centerlines") < tail.index("Could not derive")
    assert failure_tail("") == "(no output)"


def test_listed_items_exit_codes_follow_the_outcome() -> None:
    from mapsnap.loc_fit import (
        EXIT_FAILED,
        EXIT_FITTED,
        EXIT_NOT_READY,
        EXIT_UNPROCESSABLE,
        EXIT_USAGE,
        listed_items_exit_code,
    )

    code = lambda **k: listed_items_exit_code(
        **{"done": 0, "skipped": 0, "unprocessable": 0, "waiting": 0, "failed": 0, **k}
    )
    assert code(done=1) == EXIT_FITTED
    assert code(skipped=1) == EXIT_FITTED, "already done for this run tag is success"
    assert code(unprocessable=1) == EXIT_UNPROCESSABLE
    assert code(waiting=1) == EXIT_NOT_READY
    assert code(failed=1) == EXIT_FAILED
    assert code(failed=1, unprocessable=1) == EXIT_FAILED, "a crash outranks a skip"
    assert code() == EXIT_USAGE, "nothing happened at all"


def test_items_from_list_takes_the_batch_array_index(
    monkeypatch, tmp_path: Path
) -> None:
    import pytest

    from mapsnap.loc_fit import items_from_list, read_item_list, select_item

    items = [Item(f"sanborn{n}", "alabama", "1900") for n in range(3)]
    listing = tmp_path / "items.txt"
    listing.write_text("sanborn2\n\nsanborn0\n")
    assert read_item_list(str(listing), tmp_path) == ["sanborn2", "sanborn0"]
    assert [i.item for i in items_from_list(items, str(listing), 1, tmp_path)] == [
        "sanborn0"
    ]
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "0")
    assert [i.item for i in items_from_list(items, str(listing), None, tmp_path)] == [
        "sanborn2"
    ]
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX")
    with pytest.raises(SystemExit, match="item-index"):
        items_from_list(items, str(listing), None, tmp_path)
    with pytest.raises(SystemExit, match="outside the list"):
        items_from_list(items, str(listing), 7, tmp_path)
    with pytest.raises(SystemExit, match="not in the manifest"):
        select_item(items, "sanborn99")


def test_items_from_list_slices_the_list_for_a_chunked_child(tmp_path) -> None:
    import pytest

    from mapsnap.loc_fit import items_from_list

    listing = tmp_path / "items.txt"
    listing.write_text("\n".join(f"sanborn{n}" for n in range(10)) + "\n")
    items = [Item(f"sanborn{n}", "alabama", "1900") for n in range(10)]
    took = lambda index: [
        i.item for i in items_from_list(items, str(listing), index, tmp_path, 4)
    ]
    assert took(0) == ["sanborn0", "sanborn1", "sanborn2", "sanborn3"]
    assert took(1) == ["sanborn4", "sanborn5", "sanborn6", "sanborn7"]
    # The last child of an array takes the short remainder.
    assert took(2) == ["sanborn8", "sanborn9"]
    with pytest.raises(SystemExit):
        items_from_list(items, str(listing), 3, tmp_path, 4)
    with pytest.raises(SystemExit):
        items_from_list(items, str(listing), 0, tmp_path, 0)


def test_chunk_exit_code_lets_a_mixed_chunk_succeed() -> None:
    from mapsnap.loc_fit import (
        EXIT_FAILED,
        EXIT_FITTED,
        EXIT_NOT_READY,
        EXIT_UNPROCESSABLE,
        listed_items_exit_code,
    )

    code = lambda **k: listed_items_exit_code(
        **{"done": 0, "skipped": 0, "unprocessable": 0, "waiting": 0, "failed": 0, **k}
    )
    # One item missing from the mirror does not condemn the chunk that fitted
    # the other three; Batch never retries a 3, so it must mean "nothing ran".
    assert code(done=3, unprocessable=1) == EXIT_FITTED
    assert code(unprocessable=4) == EXIT_UNPROCESSABLE
    # Anything that failed asks for the retry, which re-runs the whole chunk;
    # the items already published under the run tag are skipped on the way past.
    assert code(done=3, failed=1) == EXIT_FAILED
    # An item still awaiting CRAFT leaves the chunk incomplete.
    assert code(done=2, waiting=1) == EXIT_NOT_READY


def test_balance_items_evens_out_the_chunks_and_keeps_every_item() -> None:
    from mapsnap.loc_fit import balance_items

    sheets = {"big1": 100, "big2": 90, "big3": 80} | {f"small{n}": 1 for n in range(9)}
    names = sorted(sheets)
    planned = balance_items(names, sheets, 4)
    assert sorted(planned) == sorted(names)
    chunks = [planned[i : i + 4] for i in range(0, len(planned), 4)]
    loads = [sum(sheets[n] for n in c) for c in chunks]
    # List order would put all three big volumes in one child; spreading them
    # one to a child is the whole point.
    assert all(sum(1 for n in c if sheets[n] > 50) == 1 for c in chunks), chunks
    assert max(loads) - min(loads) <= 20, loads


def test_balance_items_charges_a_short_item_for_its_fixed_cost() -> None:
    from mapsnap.loc_fit import balance_items

    # Without the fixed-cost weight every one-sheet item looks free and they
    # all pile into one child, which then pays 8 container starts back to back.
    sheets = {f"tiny{n}": 1 for n in range(8)} | {"mid1": 4, "mid2": 4}
    planned = balance_items(sorted(sheets), sheets, 5)
    chunks = [planned[i : i + 5] for i in range(0, len(planned), 5)]
    assert all(len(c) == 5 for c in chunks)
    assert sorted(planned) == sorted(sheets)


def test_balance_items_handles_a_short_final_chunk() -> None:
    from mapsnap.loc_fit import balance_items

    sheets = {f"item{n}": n for n in range(1, 8)}
    planned = balance_items(sorted(sheets), sheets, 3)
    assert sorted(planned) == sorted(sheets)
    assert len(planned) == 7


def test_count_sheets_counts_rows_per_item(tmp_path) -> None:
    from mapsnap.loc_fit import count_sheets

    manifest = tmp_path / "mapping.tsv"
    manifest.write_text(
        "item\tstate\tyear\n"
        "sanborn1\tohio\t1950\n"
        "sanborn1\tohio\t1950\n"
        "sanborn2\tohio\t1950\n"
    )
    assert count_sheets(manifest) == {"sanborn1": 2, "sanborn2": 1}
