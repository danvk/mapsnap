"""Tests for the corpus GPU pass driver (mapsnap.loc_craft)."""

import subprocess
from collections import Counter
from pathlib import Path

import pytest

from mapsnap.loc_craft import (
    Item,
    bucket_name,
    format_duration,
    key_prefix,
    list_prefix,
    plan_item,
    prepare_next,
    read_manifest,
    resolve_manifest,
    run_aws,
    select_shard,
    shard_of,
)


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch) -> None:
    """Skip the retry sleeps; without this the failure tests cost 21 s each."""
    from mapsnap import loc_craft

    monkeypatch.setattr(loc_craft.time, "sleep", lambda seconds: None)


MANIFEST = """item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir
sanborn00001_003\talabama\t1924\tabbeville\t1\t00001_1924-0001\tp1\tjp2\t9\tgmd/x
sanborn00001_003\talabama\t1924\tabbeville\t2\t00001_1924-0002\tp2\tjp2\t9\tgmd/x
sanborn05791_007\tnew-york\t1906\tbrooklyn\t1\t05791_06_1906-0001\tp1\tjp2\t9\tgmd/y
"""


def write_manifest(tmp_path: Path, text: str = MANIFEST) -> Path:
    path = tmp_path / "mapping.tsv"
    path.write_text(text)
    return path


def test_read_manifest_keeps_one_row_per_item(tmp_path: Path) -> None:
    items = read_manifest(write_manifest(tmp_path))
    assert [item.item for item in items] == ["sanborn00001_003", "sanborn05791_007"]
    assert items[0].prefix == "by-state/alabama/1924/sanborn00001_003"
    assert items[1].prefix == "by-state/new-york/1906/sanborn05791_007"


def test_read_manifest_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, "item\tcity\nx\ty\n")
    with pytest.raises(SystemExit, match="'state'"):
        read_manifest(path)


def test_shards_are_stable_disjoint_and_roughly_even() -> None:
    items = [f"sanborn{index:05d}_001" for index in range(4000)]
    counts = Counter(shard_of(item, 8) for item in items)
    assert set(counts) == set(range(8))
    assert max(counts.values()) < 2 * min(counts.values())
    # Stable across calls (and so across processes: sha1, not hash()).
    assert [shard_of(item, 8) for item in items[:20]] == [
        shard_of(item, 8) for item in items[:20]
    ]
    assert all(0 <= shard_of(item, 1) < 1 for item in items)


def test_select_shard_partitions_every_item_exactly_once(tmp_path: Path) -> None:
    items = read_manifest(write_manifest(tmp_path))
    selected = [
        item.item for shard in range(4) for item in select_shard(items, shard, 4)
    ]
    assert sorted(selected) == sorted(item.item for item in items)


ITEM = Item(item="sanborn00001_003", state="alabama", year="1924")


def test_plan_item_finds_the_images_and_the_missing_sidecars() -> None:
    work = plan_item(ITEM, ["metadata.json", "p1.jpg", "p2.jpg", "raw/p1.jpg"])
    assert work.pages == ["p1.jpg", "p2.jpg"]
    assert work.raw_sheets == ["raw/p1.jpg"]
    assert work.missing == [
        "p1.boxes.json",
        "p1.roadprob.jpg",
        "p2.boxes.json",
        "p2.roadprob.jpg",
        "raw/p1.boxes.json",
    ]
    assert not work.complete


def test_plan_item_calls_a_finished_item_complete() -> None:
    work = plan_item(
        ITEM,
        [
            "metadata.json",
            "p1.jpg",
            "p1.boxes.json",
            "p1.roadprob.jpg",
            "raw/p1.jpg",
            "raw/p1.boxes.json",
        ],
    )
    assert work.complete
    assert work.missing == []


def test_plan_item_does_not_mistake_a_sidecar_for_a_page() -> None:
    work = plan_item(ITEM, ["p1.jpg", "p1.roadprob.jpg", "p1.boxes.json"])
    assert work.pages == ["p1.jpg"]
    assert work.complete


def test_plan_item_wants_no_road_map_for_a_raw_key_map_sheet() -> None:
    work = plan_item(ITEM, ["raw/p0.jpg", "raw/p0.boxes.json"])
    assert work.pages == []
    assert work.complete


def test_list_prefix_returns_keys_relative_to_the_prefix(monkeypatch) -> None:
    prefix = "by-state/alabama/1924/sanborn00001_003"
    listing = "\t".join(
        f"{prefix}/{name}" for name in ("metadata.json", "p1.jpg", "raw/p1.jpg")
    )
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert list_prefix("s3://bucket/", prefix) == [
        "metadata.json",
        "p1.jpg",
        "raw/p1.jpg",
    ]
    assert captured["command"][:3] == ["aws", "s3api", "list-objects-v2"]
    assert "bucket" in captured["command"]
    assert f"{prefix}/" in captured["command"]


def test_list_prefix_reads_an_unmirrored_item_as_empty_not_as_a_failure(
    monkeypatch,
) -> None:
    """`aws s3 ls` could not tell these apart; s3api says exit 0 and "None"."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="None\n", stderr=""
        ),
    )
    assert list_prefix("s3://bucket", "by-state/x/1900/never-mirrored") == []


def test_list_prefix_strips_a_path_in_the_bucket_url(monkeypatch) -> None:
    """A bucket URL with a path once made every item look complete (no work done)."""
    prefix = "by-state/alabama/1924/sanborn00001_003"
    listing = f"_craft/selftest/{prefix}/p1.jpg"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=listing, stderr=""
        ),
    )
    assert (
        key_prefix("s3://bucket/_craft/selftest", prefix) == f"_craft/selftest/{prefix}"
    )
    assert key_prefix("s3://bucket", prefix) == prefix
    assert list_prefix("s3://bucket/_craft/selftest", prefix) == ["p1.jpg"]
    assert bucket_name("s3://bucket/_craft/selftest") == "bucket"
    assert bucket_name("s3://bucket") == "bucket"


def test_list_prefix_raises_when_the_cli_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Access Denied\n"
        ),
    )
    with pytest.raises(OSError, match="Access Denied"):
        list_prefix("s3://bucket", "by-state/x")


def test_resolve_manifest_prefers_a_local_path_and_downloads_otherwise(
    tmp_path: Path, monkeypatch
) -> None:
    local = write_manifest(tmp_path)
    assert resolve_manifest(str(local), "s3://bucket", tmp_path) == local

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (tmp_path / "work" / "loc-sanborn-maps.mapping.tsv").write_text(MANIFEST)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = resolve_manifest(None, "s3://bucket/", tmp_path / "work")
    assert path == tmp_path / "work" / "loc-sanborn-maps.mapping.tsv"
    assert calls[0][:3] == ["aws", "s3", "cp"]
    assert calls[0][3] == "s3://bucket/loc-sanborn-maps.mapping.tsv"

    # Already downloaded: no second fetch.
    calls.clear()
    resolve_manifest(None, "s3://bucket/", tmp_path / "work")
    assert calls == []


def test_format_duration_reads_as_hours_and_minutes() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(3600) == "1:00"
    assert format_duration(3660) == "1:01"
    assert format_duration(258000) == "71:40"


def test_limit_counts_work_done_not_items_skipped(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A pilot re-run must process --limit fresh items, not stop after N skips."""
    from mapsnap import loc_craft

    manifest = write_manifest(
        tmp_path,
        MANIFEST + "sanborn00009_004\talabama\t1930\tdothan\t1\ts\tp1\tjp2\t9\tgmd/z\n",
    )
    # Everything the worker reaches before the last item is already finished, so
    # the skips come first whatever order the shuffle picks.
    order = [item.item for item in select_shard(read_manifest(manifest), 0, 1)]
    done, pending = order[:-1], order[-1]
    listings = {name: ["p1.jpg", "p1.boxes.json", "p1.roadprob.jpg"] for name in done}
    listings[pending] = ["p1.jpg"]
    monkeypatch.setattr(
        loc_craft, "list_prefix", lambda bucket, prefix: listings[prefix.split("/")[-1]]
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mapsnap loc-craft",
            "--manifest",
            str(manifest),
            "--work-dir",
            str(tmp_path / "work"),
            "--dry-run",
            "--limit",
            "1",
        ],
    )
    loc_craft.main()
    out = capsys.readouterr()
    assert pending in out.out
    assert f"1 items processed, {len(done)} already complete" in out.err


def test_run_aws_retries_a_transient_failure_then_succeeds(monkeypatch) -> None:
    """The first S3 call of a fresh instance can beat its instance credentials."""
    attempts = []

    def fake_run(command, **kwargs):
        attempts.append(command)
        code = 0 if len(attempts) == 3 else 1
        return subprocess.CompletedProcess(command, code, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_aws(["aws", "s3", "ls", "s3://b/x"]).stdout == "ok"
    assert len(attempts) == 3


def test_run_aws_gives_up_with_the_exit_status_when_there_is_no_message(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, stdout="", stderr=""
        ),
    )
    with pytest.raises(OSError, match="exit 2.*no output"):
        run_aws(["aws", "s3", "ls", "s3://b/x"])


def test_shard_order_is_shuffled_but_reproducible() -> None:
    """--limit must sample a representative mix, not the oldest item ids."""
    items = [
        Item(item=f"sanborn{index:05d}_001", state="x", year="1900")
        for index in range(200)
    ]
    order = [item.item for item in select_shard(items, 0, 1)]
    assert order == [item.item for item in select_shard(items, 0, 1)]  # reproducible
    assert order != [item.item for item in items]  # and not id order
    assert sorted(order) == sorted(item.item for item in items)  # nothing lost
    assert order != [item.item for item in select_shard(items, 0, 1, seed=7)]


def test_prepare_next_walks_past_finished_items_and_records_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """The scan returns the first item needing work, and what it passed to reach it."""
    from mapsnap import loc_craft

    items = [
        Item(item=name, state="x", year="1900")
        for name in ("done1", "broken", "done2", "needs-work", "later")
    ]
    complete = ["p1.jpg", "p1.boxes.json", "p1.roadprob.jpg"]

    def fake_list(bucket, prefix):
        name = prefix.split("/")[-1]
        if name == "broken":
            raise OSError("aws s3 ls failed after 4 attempts")
        return complete if name.startswith("done") else ["p1.jpg"]

    fetched = []
    monkeypatch.setattr(loc_craft, "list_prefix", fake_list)
    monkeypatch.setattr(
        loc_craft,
        "fetch_item",
        lambda work, bucket, dir: fetched.append(work.item.item) or dir,
    )
    pending = iter(list(enumerate(items, start=1)))
    first = prepare_next(pending, "s3://b", tmp_path)
    assert first.work is not None and first.work.item.item == "needs-work"
    assert first.index == 4
    assert first.skipped == 2
    assert [name for name, _ in first.failures] == ["broken"]
    assert fetched == ["needs-work"]

    # The iterator is shared, so the next call continues where this one stopped.
    second = prepare_next(pending, "s3://b", tmp_path)
    assert second.work is not None and second.work.item.item == "later"

    # Exhausted: no work, and nothing left to report.
    third = prepare_next(pending, "s3://b", tmp_path)
    assert third.work is None and third.skipped == 0 and third.failures == []


def test_prepare_next_can_skip_the_download(tmp_path: Path, monkeypatch) -> None:
    """--dry-run plans items without pulling a byte."""
    from mapsnap import loc_craft

    monkeypatch.setattr(loc_craft, "list_prefix", lambda bucket, prefix: ["p1.jpg"])
    monkeypatch.setattr(
        loc_craft, "fetch_item", lambda *a: pytest.fail("should not download")
    )
    prepared = prepare_next(
        iter([(1, Item(item="x", state="s", year="1900"))]),
        "s3://b",
        tmp_path,
        fetch=False,
    )
    assert prepared.work is not None


def test_an_unmirrored_item_is_counted_apart_from_a_finished_one(
    tmp_path: Path, monkeypatch
) -> None:
    """45 manifest items were never mirrored; they are absent, not complete."""
    from mapsnap import loc_craft

    items = [Item(item=name, state="x", year="1900") for name in ("gone", "needs")]
    monkeypatch.setattr(
        loc_craft,
        "list_prefix",
        lambda bucket, prefix: [] if prefix.endswith("gone") else ["p1.jpg"],
    )
    monkeypatch.setattr(loc_craft, "fetch_item", lambda work, bucket, dir: dir)
    prepared = prepare_next(iter(list(enumerate(items, start=1))), "s3://b", tmp_path)
    assert prepared.work is not None and prepared.work.item.item == "needs"
    assert prepared.absent == 1
    assert prepared.skipped == 0
    assert prepared.failures == []


# The queue as an item source (#354)

ALPHA = Item("sanborn1", "alabama", "1900")
BETA = Item("sanborn2", "alabama", "1901")


def _queue_source(monkeypatch, bodies: list[str], items: dict[str, Item]):
    """A QueueSource backed by a scripted queue, plus the deletes it performs."""
    from mapsnap import work_queue
    from mapsnap.loc_craft import QueueSource

    pending = [work_queue.Message(body, f"handle-{body}") for body in bodies]
    deleted: list[str] = []
    monkeypatch.setattr(
        work_queue, "receive", lambda url, **kw: [pending.pop(0)] if pending else []
    )
    monkeypatch.setattr(
        work_queue, "delete", lambda url, handle: deleted.append(handle)
    )
    return QueueSource("https://q", items), deleted


def test_queue_source_yields_manifest_items_until_drained(monkeypatch) -> None:
    source, deleted = _queue_source(
        monkeypatch, ["sanborn1", "sanborn2"], {"sanborn1": ALPHA, "sanborn2": BETA}
    )
    assert [(i, it.item) for i, it in source] == [(1, "sanborn1"), (2, "sanborn2")]
    assert deleted == []  # nothing is retired merely by being taken


def test_queue_source_drops_a_name_the_manifest_does_not_have(monkeypatch) -> None:
    """Otherwise it is redelivered forever and the queue never drains."""
    source, deleted = _queue_source(
        monkeypatch, ["ghost", "sanborn1"], {"sanborn1": ALPHA}
    )
    assert [it.item for _, it in source] == ["sanborn1"]
    assert deleted == ["handle-ghost"]


def test_queue_source_retire_deletes_and_release_does_not(monkeypatch) -> None:
    source, deleted = _queue_source(
        monkeypatch, ["sanborn1", "sanborn2"], {"sanborn1": ALPHA, "sanborn2": BETA}
    )
    taken = [it for _, it in source]
    source.retire(taken[0])
    assert deleted == ["handle-sanborn1"]
    # A failure leaves the lease to lapse so another worker retries it.
    source.release(taken[1])
    assert deleted == ["handle-sanborn1"]
    source.retire(taken[1])  # already released: nothing left to delete
    assert deleted == ["handle-sanborn1"]


def test_prepare_next_retires_what_it_settles_itself(monkeypatch) -> None:
    """Skipped and absent items must leave the queue; failures must not."""
    from mapsnap import loc_craft

    complete = Item("done", "alabama", "1900")
    absent = Item("gone", "alabama", "1901")
    retired: list[str] = []

    monkeypatch.setattr(
        loc_craft,
        "list_prefix",
        lambda bucket, prefix: [] if prefix.endswith("gone") else ["p1.jpg"],
    )
    monkeypatch.setattr(
        loc_craft,
        "plan_item",
        lambda item, present: loc_craft.ItemWork(item, [], [], []),
    )
    prepared = loc_craft.prepare_next(
        iter([(1, complete), (2, absent)]),
        "s3://b",
        Path("/tmp"),
        fetch=False,
        retire=lambda item: retired.append(item.item),
    )
    assert prepared.work is None
    assert (prepared.skipped, prepared.absent) == (1, 1)
    assert retired == ["done", "gone"]


def test_prepare_next_leaves_a_failed_item_for_another_worker(monkeypatch) -> None:
    """A listing failure is transient; retiring it would lose the item."""
    from mapsnap import loc_craft

    def boom(bucket, prefix):
        raise OSError("listing failed")

    retired: list[str] = []
    monkeypatch.setattr(loc_craft, "list_prefix", boom)
    prepared = loc_craft.prepare_next(
        iter([(1, Item("flaky", "alabama", "1900"))]),
        "s3://b",
        Path("/tmp"),
        fetch=False,
        retire=lambda item: retired.append(item.item),
    )
    assert [name for name, _ in prepared.failures] == ["flaky"]
    assert retired == []
