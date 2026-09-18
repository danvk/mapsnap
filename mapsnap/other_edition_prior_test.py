import json
from pathlib import Path

import pytest

from mapsnap.other_edition_prior import (
    OTHER_EDITION_CONTRADICTION_M,
    OTHER_EDITION_RADIUS_M,
    OtherEditionPrior,
    annotation_centers,
    annotation_signature,
    ensure_prior,
    item_center,
    load_prior,
    other_edition_plan,
    prior_path,
    section_key,
)

LAT0 = 41.9
DEG_PER_M_LON = 1.0 / (111_320.0 * 0.745)  # cos(41.9 deg)
WIDTH, HEIGHT = 1000, 1300


def item(
    stem: str, lon: float, lat: float = LAT0, transform: str = "polynomial"
) -> dict:
    """One annotated page, 300 m x 400 m, centered on (lon, lat).

    The GCPs are the page's own corners, so a fit of them reproduces the
    intended placement exactly.
    """
    half_lon, half_lat = 150 * DEG_PER_M_LON, 200 / 110_540.0
    corners = [
        ((0, 0), (lon - half_lon, lat + half_lat)),
        ((WIDTH, 0), (lon + half_lon, lat + half_lat)),
        ((WIDTH, HEIGHT), (lon + half_lon, lat - half_lat)),
        ((0, HEIGHT), (lon - half_lon, lat - half_lat)),
    ]
    return {
        "label": f"Chicago, Ill. | 1950 | Vol. 1 {stem}",
        "body": {
            "transformation": {"type": transform},
            "features": [
                {
                    "properties": {"resourceCoords": [px, py]},
                    "geometry": {"coordinates": [world_lon, world_lat]},
                }
                for (px, py), (world_lon, world_lat) in corners
            ],
        },
        "target": {"source": {"id": None, "width": WIDTH, "height": HEIGHT}},
    }


def annotation(path: Path, items: list[dict]) -> Path:
    """Write a IIIF AnnotationPage holding these items."""
    path.write_text(json.dumps({"type": "AnnotationPage", "items": items}))
    return path


def other_edition_annotation(
    tmp_path: Path, count: int = 6
) -> tuple[Path, list[float]]:
    """An annotation placing sheets p1..pN on a west-to-east line, and their lons."""
    lons = [-87.60 - 0.004 * i for i in range(count)]
    path = annotation(
        tmp_path / "sib.iiif.json",
        [item(f"p{i + 1}", lon) for i, lon in enumerate(lons)],
    )
    return path, lons


def test_section_key_drops_division_letter_and_keeps_sheet_letter():
    assert section_key("p10n") == "P10"
    assert section_key("p55w") == "P55"
    assert section_key("p10") == "P10"
    assert section_key("p3a") == "P3A"
    assert section_key("p12n__1") == ""
    # Not a numbered sheet: a cover page, and the path-shaped junk
    # source_id_to_page_key returns for an image URL it cannot parse.
    assert section_key("covr") == ""
    assert section_key("data/chicago_il_1906_vol_1/p1.jpg") == ""
    # A compound suffix is ambiguous — a skeleton twin of p6n, or a sequence
    # letter on p6 — so it keys as nothing rather than as a guess.
    assert section_key("p6ns") == ""
    assert section_key("p12ws") == ""
    assert section_key("p6ns__1") == ""


def test_the_center_is_the_page_rectangle_not_the_annotated_region():
    """The selector is a sub-polygon of the page; its centroid is tens of metres off."""
    page = item("p1", -87.6)
    # A selector covering only the top-left quarter. It must not be read.
    page["target"]["selector"] = {
        "type": "SvgSelector",
        "value": '<svg><polygon points="0,0 500,0 500,650 0,650" /></svg>',
    }
    center = item_center(page)
    assert center is not None
    assert abs(center[0] + 87.6) < 1e-9
    assert abs(center[1] - LAT0) < 1e-9


def test_an_item_without_a_fittable_transform_has_no_center():
    page = item("p1", -87.6)
    page["body"]["features"] = page["body"]["features"][:2]  # too few for an affine
    assert item_center(page) is None
    # A helmert annotation fits from two.
    helmert = item("p1", -87.6, transform="helmert")
    helmert["body"]["features"] = helmert["body"]["features"][:2]
    assert item_center(helmert) is not None
    # No page dimensions to place a rectangle on.
    sized = item("p1", -87.6)
    sized["target"]["source"] = {"id": None}
    assert item_center(sized) is None


def test_annotation_centers_keys_by_sheet_and_skips_panels(tmp_path: Path):
    path, lons = other_edition_annotation(tmp_path)
    # A division letter is dropped, a split panel gets no key at all.
    extra = [item("p9N", -87.70), item("p9N [2]", -87.30)]
    path = annotation(
        tmp_path / "sib.iiif.json", json.loads(path.read_text())["items"] + extra
    )
    centers = annotation_centers(path)
    assert set(centers) == {"P1", "P2", "P3", "P4", "P5", "P6", "P9"}
    assert abs(centers["P3"][0] - lons[2]) < 1e-9
    assert abs(centers["P9"][0] + 87.70) < 1e-9


def test_two_labels_keying_alike_never_resolve_by_file_order(tmp_path: Path):
    """A silent overwrite here would seed a 50 m search half a sheet away."""
    # 'p83' and the skeleton 'p83s' share the key P83: the stem that IS the
    # key wins, whichever order the annotation happens to list them in.
    for order in (
        [item("p83", -87.60), item("p83s", -87.90)],
        [item("p83s", -87.90), item("p83", -87.60)],
    ):
        centers = annotation_centers(annotation(tmp_path / "skeleton.iiif.json", order))
        assert abs(centers["P83"][0] + 87.60) < 1e-9

    # Two division letters and no bare sheet: the key names no single sheet,
    # so it is dropped rather than resolved arbitrarily. Three read the same.
    both = annotation(
        tmp_path / "divided.iiif.json", [item("p5N", -87.60), item("p5S", -87.90)]
    )
    assert "P5" not in annotation_centers(both)
    three = annotation(
        tmp_path / "three.iiif.json",
        [item("p7N", -87.60), item("p7S", -87.70), item("p7W", -87.80)],
    )
    assert "P7" not in annotation_centers(three)


def test_an_annotation_that_keys_nothing_is_refused(tmp_path: Path):
    """`mapsnap iiif --image-base-url` writes ids no page number can be read from.

    The centers then key as filesystem paths, match no sheet, and the run would
    otherwise look like an edition that simply did not help.
    """
    page = item("p1", -87.6)
    page["target"]["source"]["id"] = "data/chicago_il_1906_vol_1/p1.jpg"
    path = annotation(tmp_path / "unkeyable.iiif.json", [page])
    assert annotation_centers(path) == {}
    with pytest.raises(SystemExit) as raised:
        ensure_prior(tmp_path / "volume", path)
    assert "no sheet centers" in str(raised.value)
    assert "p1.jpg" in str(raised.value)


def test_the_prior_answers_for_a_sheet_and_only_a_sheet(tmp_path: Path):
    path, lons = other_edition_annotation(tmp_path)
    prior = ensure_prior(tmp_path / "volume", path)
    assert prior.center_for("p3n") is not None
    assert abs(prior.center_for("p3n")[0] - lons[2]) < 1e-9  # type: ignore[index]
    assert prior.center_for("p3n__1") is None  # a panel is not a sheet
    assert prior.center_for("p99n") is None  # the other edition never placed it
    assert "6 sheets" in prior.describe()


def test_the_signature_follows_the_annotation_content(tmp_path: Path):
    path, lons = other_edition_annotation(tmp_path)
    before = annotation_signature(path)
    # A copy of the same bytes is the same edition, whatever its timestamp.
    copy = annotation(
        tmp_path / "copy.iiif.json", json.loads(path.read_text())["items"]
    )
    assert annotation_signature(copy) == before
    # A moved placement is not.
    annotation(path, [item(f"p{i + 1}", lon) for i, lon in enumerate(lons)][:5])
    assert annotation_signature(path) != before


def test_the_prior_round_trips_through_the_sidecar_it_is_cached_in(tmp_path: Path):
    path, _ = other_edition_annotation(tmp_path)
    volume = tmp_path / "volume"
    prior = ensure_prior(volume, path)
    assert prior.signature == annotation_signature(path)
    sidecar = prior_path(volume)
    assert sidecar == volume / "artifacts" / "osm_snap" / "other_edition_prior.json"
    assert load_prior(volume) == prior

    # A key an older run wrote is ignored, not a load failure.
    doc = json.loads(sidecar.read_text())
    doc["radius_m"] = 150.0
    sidecar.write_text(json.dumps(doc))
    assert load_prior(volume) == prior
    assert load_prior(tmp_path / "nowhere") is None


def prior_at(lon: float = -87.6, lat: float = LAT0) -> OtherEditionPrior:
    """A prior placing sheet P1 at (lon, lat)."""
    return OtherEditionPrior(
        source="other-edition.iiif.json",
        centers={"P1": (lon, lat)},
        signature="sig",
    )


def plan_for(stem: str = "p1n", keymap: tuple[float, float] | None = None, **kwargs):
    """other_edition_plan for one page, with or without a key-map center."""
    regions = None if keymap is None else [[[keymap[0], keymap[1]]]]
    return other_edition_plan(
        kwargs.pop("prior", prior_at()),
        stem,
        centers=[] if keymap is None else [keymap],
        regions=regions,
        radius_m=250.0,
        rescued=kwargs.pop("rescued", True),
        **kwargs,
    )


def test_the_plan_replaces_a_key_map_the_other_edition_contradicts():
    """A key map 8 km out is wrong: the other edition's center stands alone."""
    plan = plan_for(keymap=(-87.7, 41.8))
    assert plan.replaced_key_map
    assert (plan.sheet_center, plan.centers, plan.regions) == (
        (-87.6, LAT0),
        [(-87.6, LAT0)],
        None,
    )
    assert plan.radius_m == OTHER_EDITION_RADIUS_M  # not the volume's 250 m
    # A page with no key-map center at all is the same case.
    assert plan_for().replaced_key_map


def test_the_prior_leaves_a_key_map_that_agrees_alone():
    """The Chicago 1919 vol 22 case: a good key map keeps its centers and window."""
    keymap = (-87.6, LAT0 - 100.0 / 110_540.0)  # 100 m south, inside the bar
    plan = plan_for(keymap=keymap)
    assert not plan.replaced_key_map
    assert plan.sheet_center == (-87.6, LAT0)  # known, but not used
    assert (plan.centers, plan.regions) == ([keymap], [[[keymap[0], keymap[1]]]])
    assert plan.radius_m == 250.0


def test_the_disagreement_bar_is_the_contradiction_threshold():
    """Which side of OTHER_EDITION_CONTRADICTION_M a key map falls on decides it."""
    from mapsnap.utils import haversine_m

    # Calibrated against the same haversine the rule uses, so a 1 m test
    # margin is not eaten by a hand-written degrees-per-metre constant.
    deg_per_m = 1e-4 / haversine_m(LAT0, -87.6, LAT0 + 1e-4, -87.6)

    def replaced_at(distance_m: float) -> bool:
        plan = plan_for(keymap=(-87.6, LAT0 - distance_m * deg_per_m))
        return plan.replaced_key_map

    assert OTHER_EDITION_CONTRADICTION_M == 200.0
    assert not replaced_at(OTHER_EDITION_CONTRADICTION_M - 1.0)
    assert replaced_at(OTHER_EDITION_CONTRADICTION_M + 1.0)


def test_the_plan_leaves_every_other_page_alone():
    keymap = (-87.7, 41.8)

    def untouched(plan) -> bool:
        return (
            plan.sheet_center,
            plan.replaced_key_map,
            plan.centers,
            plan.radius_m,
        ) == (
            None,
            False,
            [keymap],
            250.0,
        )

    # A page that already has a pose, a split panel, a sheet the other
    # edition never placed, and no other edition at all.
    assert untouched(plan_for(keymap=keymap, rescued=False))
    assert untouched(plan_for("p1n__1", keymap=keymap))
    assert untouched(plan_for("p2n", keymap=keymap))
    assert untouched(plan_for(keymap=keymap, prior=None))
