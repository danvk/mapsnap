from pathlib import Path

import pytest

from mapsnap.download_loc_iiif import (
    CanvasTarget,
    canvas_to_page_key,
    disambiguate_keys,
    select_pages,
)

LOC = (
    "https://tile.loc.gov/image-services/iiif/service:gmd:gmd411m:g4114gm:g04023195307"
)


def target(segment: str, label: str = "Page", out: str = "/vol") -> CanvasTarget:
    canvas_id = f"{LOC}:{segment}"
    key = canvas_to_page_key(canvas_id, label)
    return CanvasTarget({"@id": canvas_id, "label": label}, Path(out), key, key)


def test_canvas_to_page_key():
    assert canvas_to_page_key(f"{LOC}:04023_07_1953-0701", "Page 7") == "p701"
    # Page 0 of any sub-volume reduces to the same key — the collision this
    # module has to disambiguate.
    assert canvas_to_page_key(f"{LOC}:04023_07_1953-0000", "Page 6") == "p0"
    assert canvas_to_page_key(f"{LOC}:04023_08_1953-0000", "Page 36") == "p0"
    # Front matter keeps its whole segment, so it never collides.
    assert (
        canvas_to_page_key(f"{LOC}:04023_07_1953-titl", "Title") == "04023_07_1953-titl"
    )


def test_disambiguate_leaves_unique_keys_alone():
    targets = [target("04023_07_1953-0701"), target("04023_07_1953-0702")]
    assert disambiguate_keys(targets) == []
    assert [t.page_key for t in targets] == ["p701", "p702"]


def test_disambiguate_suffixes_collisions_in_manifest_order():
    targets = [
        target("04023_07_1953-0000", "Page 6"),  # volume 07's key map
        target("04023_07_1953-0701"),
        target("04023_08_1953-0000", "Page 36"),  # volume 08's key map
    ]
    assert disambiguate_keys(targets) == [("p0", ["p0a", "p0b"])]
    assert [t.page_key for t in targets] == ["p0a", "p701", "p0b"]
    # The base key is retained so a --pages selection can still name it.
    assert [t.base_key for t in targets] == ["p0", "p701", "p0"]


def test_disambiguate_skips_a_suffix_another_canvas_already_owns():
    # A manifest with a real "p0a" must not have a collision renamed onto it.
    targets = [
        target("04023_07_1953-0000", "Page 6"),
        target("04023_08_1953-0000", "Page 36"),
        target("04023_09_1953-000a", "Page 60"),
    ]
    assert [t.base_key for t in targets] == ["p0", "p0", "p0a"]
    disambiguate_keys(targets)
    assert [t.page_key for t in targets] == ["p0b", "p0c", "p0a"]


def test_disambiguate_is_per_output_directory():
    # The same key in two volumes is not a collision: different files.
    targets = [
        target("04023_07_1953-0000", out="/vol7"),
        target("04023_08_1953-0000", out="/vol8"),
    ]
    assert disambiguate_keys(targets) == []
    assert [t.page_key for t in targets] == ["p0", "p0"]


def test_disambiguate_rejects_more_than_26_collisions():
    targets = [target(f"04023_{i:02d}_1953-0000") for i in range(27)]
    with pytest.raises(ValueError, match="more than 26"):
        disambiguate_keys(targets)


def test_select_pages_matches_final_or_base_key():
    targets = [
        target("04023_07_1953-0000", "Page 6"),
        target("04023_08_1953-0000", "Page 36"),
        target("04023_07_1953-0701"),
    ]
    disambiguate_keys(targets)
    # The final key names one variant...
    assert [t.page_key for t in select_pages(targets, {"p0a"})] == ["p0a"]
    # ...and the base key names all of them, without having to know the suffixes.
    assert [t.page_key for t in select_pages(targets, {"p0"})] == ["p0a", "p0b"]
    assert [t.page_key for t in select_pages(targets, {"p701"})] == ["p701"]
    assert select_pages(targets, {"p999"}) == []


def test_target_path_and_url():
    t = target("04023_07_1953-0701")
    assert t.path == Path("/vol/p701.jpg")
    assert t.image_url("full").endswith("-0701/full/full/0/default.jpg")
    assert "pct:25" in t.image_url("pct:25")


def test_find_target_matches_the_disambiguated_key():
    from mapsnap.download_raw import find_target

    targets = [
        target("04023_07_1953-0000", "Page 6"),
        target("04023_08_1953-0000", "Page 36"),
        target("04023_07_1953-0701"),
    ]
    disambiguate_keys(targets)
    # A scaled p0a.jpg must find the FIRST page-0 canvas, not fail on "p0".
    assert find_target(targets, "p0a").label == "Page 6"
    assert find_target(targets, "p0b").label == "Page 36"
    assert find_target(targets, "p701").page_key == "p701"
    with pytest.raises(ValueError, match="No canvas found"):
        find_target(targets, "p0")
