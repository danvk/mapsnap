"""Tests for the reconcile arbitration pass (#270 v1)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mapsnap.reconcile import (
    KEEP_PRIOR,
    UNPLACED,
    Hypothesis,
    PageNode,
    build_edges,
    collect_hypotheses,
    dedupe_hypotheses,
    pairwise_energy,
    pose_scale_log2,
    published_channel,
    solve,
    unary_energy,
)

# A ~0.6 m/px north-up test frame near the equator-adjacent test latitude the
# snap tests use, so metre arithmetic is easy to reason about.
LAT0 = 40.0
KX = 111_320.0 * math.cos(math.radians(LAT0))
KY = 110_540.0
M_PER_PX = 0.6


def affine(shift_east_m: float = 0.0, shift_north_m: float = 0.0, scale: float = 1.0):
    """North-up page->world affine displaced by metres from a fixed origin."""
    s = M_PER_PX * scale
    return np.array(
        [
            [s / KX, 0.0, -74.0 + shift_east_m / KX],
            [0.0, -s / KY, LAT0 + shift_north_m / KY],
        ]
    )


def georef_doc(a: np.ndarray, width: int = 1000, height: int = 800, n_inliers: int = 4):
    """A minimal georef sidecar doc with corners derived from the affine."""
    corners = [
        [a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2]]
        for x, y in [(0, 0), (width, 0), (width, height), (0, height)]
    ]
    # Distinct pixel positions so effective_gcp_count's 60px clustering sees
    # them as separate physical intersections.
    intersections = [
        {"x": 100 + 200 * i, "y": 100 + 150 * i, "inlier": True, "lon": -74, "lat": 40}
        for i in range(n_inliers)
    ]
    return {
        "width": width,
        "height": height,
        "corners": corners,
        "streets": [],
        "intersections": intersections,
    }


def write_sidecar(tmp_path: Path, stem: str, variant: str, doc: dict) -> Path:
    path = tmp_path / f"{stem}.{variant}.json"
    path.write_text(json.dumps(doc))
    return path


def make_unit(stem: str, width: int = 1000, height: int = 800):
    from mapsnap.edge_join_experiment import PageUnit

    return PageUnit(
        stem=stem,
        number=1,
        width=width,
        height=height,
        fit_state="fitted",
        truth=None,
        split_truth=False,
        gen_affine=None,
        inlier_intersections=0,
        inlier_streets=0,
        keymap_centers=[],
        keymap_radius_m=600.0,
    )


def scored(
    source: str,
    a,
    verification: float,
    gcps: int = 2,
    published=False,
    page_placed=True,
):
    h = Hypothesis(source=source, affine=a, effective_gcps=gcps)
    if a is not None:
        h.scores["verification"] = verification
    unary_energy(h, published, 600.0, None, None, page_placed=page_placed)
    return h


def test_collect_globs_all_variants(tmp_path):
    # The p55 case: a rejected keymap-outlier pose and a published osm pose
    # must BOTH become hypotheses (GEOREF_VARIANTS omits -keymap-outlier).
    write_sidecar(tmp_path, "p55", "georef-keymap-outlier", georef_doc(affine(0)))
    write_sidecar(tmp_path, "p55", "georef-snap", georef_doc(affine(5000)))
    hypotheses, published = collect_hypotheses(tmp_path, "p55", None, None)
    sources = [h.source for h in hypotheses]
    assert "georef-keymap-outlier" in sources
    assert "georef-snap" in sources
    assert sources[-1] == UNPLACED
    assert published is not None and hypotheses[published].source == "georef-snap"


def test_published_channel_resolution_follows_glob_order(tmp_path):
    write_sidecar(tmp_path, "p1", "georef", georef_doc(affine(0)))
    write_sidecar(tmp_path, "p1", "georef-snap", georef_doc(affine(5000)))
    assert published_channel(tmp_path, "p1") == "georef-snap"
    write_sidecar(tmp_path, "p1", "georef-street", georef_doc(affine(9000)))
    assert published_channel(tmp_path, "p1") == "georef-street"


def test_dedupe_merges_near_identical_and_keeps_max_gcps():
    a = Hypothesis(source="georef", affine=affine(0), effective_gcps=1)
    b = Hypothesis(source="snap:0", affine=affine(0.5), effective_gcps=4)
    c = Hypothesis(source="snap:1", affine=affine(400), effective_gcps=0)
    kept = dedupe_hypotheses([a, b, c])
    assert [h.source for h in kept] == ["georef", "snap:1"]
    assert kept[0].merged_sources == ["snap:0"]
    assert kept[0].effective_gcps == 4


def test_unplaced_bar():
    # A junk 1-GCP pose with no P(road) support loses to UNPLACED even as the
    # published pose (fargo p63__4: verification -0.14, a bucket measuring
    # 0.224 good land against 0.224 disaster land); a real fit wins easily.
    weak = scored("georef", affine(0), verification=-0.14, gcps=1, published=True)
    strong = scored("georef", affine(0), verification=1.8, gcps=4, published=True)
    unplaced = scored(UNPLACED, None, 0.0)
    assert weak.unary > unplaced.unary
    assert strong.unary < unplaced.unary


def test_keep_prior_defends_by_measured_support():
    # The published pose's protection is its MEASURED expected value, not a
    # flat epsilon: a 4+ GCP fit whose P(road) says nothing still carries 0.57
    # (the corpus rate at which such poses are good), while a 1-GCP fit in the
    # same situation carries ~0 — that bucket is a genuine coin flip.
    strong = scored("georef", affine(0), verification=-0.45, gcps=5, published=True)
    weak = scored("georef", affine(0), verification=-0.45, gcps=1, published=True)
    unplaced = scored(UNPLACED, None, 0.0)
    assert strong.unary < unplaced.unary  # kept on its prior
    assert weak.unary > unplaced.unary  # no evidence, no prior, no keep
    assert KEEP_PRIOR[("4+", False)] > KEEP_PRIOR[("0-1", False)]


def test_pose_scale_log2_tracks_scale():
    assert pose_scale_log2(affine(scale=2.0)) - pose_scale_log2(affine()) == (
        pytest.approx(1.0, abs=1e-6)
    )


def make_node(stem: str, hypotheses, published=0, base=None):
    return PageNode(
        unit=make_unit(stem),
        is_panel=base is not None,
        base=base,
        hypotheses=hypotheses,
        published_index=published,
    )


def adjacency_for(claims: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """adjacency.json shape: claims[stem][other] = (x_frac, y_frac)."""
    pages = {}
    for stem, others in claims.items():
        detections = [
            {
                # adjacency.json records the printed page NUMBER as the key
                # ("2"), not the stem ("p2") — page_key() strips the prefix.
                "key": other.lstrip("p").upper(),
                "claim": True,
                "x_frac": frac[0],
                "y_frac": frac[1],
            }
            for other, frac in others.items()
        ]
        pages[stem] = {"detections": detections}
    return {
        "pages": pages,
        "adjacency": [["p1", "p2"], ["p2", "p3"]],
    }


def chain_adjacency() -> dict:
    # p1 | p2 | p3 left-to-right, each claiming its neighbor at the shared edge.
    return adjacency_for(
        {
            "p1": {"p2": (1.0, 0.5)},
            "p2": {"p1": (0.0, 0.5), "p3": (1.0, 0.5)},
            "p3": {"p2": (0.0, 0.5)},
        }
    )


PAGE_EAST_M = 1000 * M_PER_PX  # one page width in metres


def test_joint_beats_greedy():
    # Three pages in a row. The middle page's best UNARY hypothesis is an
    # alias one page-width east; the correct pose scores slightly lower alone
    # but agrees with both neighbors' stamps. Greedy picks the alias; the
    # joint solve must not.
    correct = scored("georef", affine(PAGE_EAST_M), verification=1.6)
    alias = scored("snap:0", affine(2 * PAGE_EAST_M), verification=1.85)
    nodes = {
        "p1": make_node(
            "p1",
            [scored("georef", affine(0), verification=1.8), scored(UNPLACED, None, 0)],
        ),
        "p2": make_node("p2", [alias, correct, scored(UNPLACED, None, 0)]),
        "p3": make_node(
            "p3",
            [
                scored("georef", affine(2 * PAGE_EAST_M), verification=1.8),
                scored(UNPLACED, None, 0),
            ],
        ),
    }
    greedy = min(range(2), key=lambda i: nodes["p2"].hypotheses[i].unary)
    assert nodes["p2"].hypotheses[greedy].source == "snap:0"
    adjacency = chain_adjacency()
    edges = build_edges(nodes, adjacency)
    assert ("stamp", "p1", "p2") in edges
    assignment = solve(nodes, edges, adjacency, 0.0, (-74.0, LAT0))
    assert nodes["p2"].hypotheses[assignment["p2"]].source == "georef"


def test_stamp_factor_scale_aware():
    # The same metre disagreement costs less between coarse sheets: parity
    # with adjacency_gate.edge_scale_factor.
    # Both pairs disagree by the same 150 m of stamp separation; the coarse
    # pages are twice as wide, so coarse p2 sits at twice the page width.
    fine_a = scored("georef", affine(0), verification=1.5)
    fine_b = scored("georef", affine(PAGE_EAST_M + 150), verification=1.5)
    coarse_a = scored("georef", affine(0, scale=2.0), verification=1.5)
    coarse_b = scored(
        "georef", affine(2 * PAGE_EAST_M + 150, scale=2.0), verification=1.5
    )
    nodes = {
        "p1": make_node("p1", [fine_a]),
        "p2": make_node("p2", [fine_b]),
    }
    adjacency = chain_adjacency()
    fine = pairwise_energy(
        "stamp",
        adjacency,
        nodes["p1"],
        fine_a,
        nodes["p2"],
        fine_b,
        0.0,
        (-74.0, LAT0),
    )
    median_log = pose_scale_log2(affine()) * math.log(2)
    coarse = pairwise_energy(
        "stamp",
        adjacency,
        nodes["p1"],
        coarse_a,
        nodes["p2"],
        coarse_b,
        median_log,
        (-74.0, LAT0),
    )
    assert coarse < fine


def test_no_sibling_factor_between_panels():
    # Split panels are separate maps sharing a sheet; they may legitimately
    # depict overlapping ground (a small inset detailing an area the large
    # panel also covers — kansas_city p526, both fits correct). No edge is
    # created between siblings, so overlap costs them nothing.
    strong = scored("georef", affine(0), verification=1.9, gcps=4)
    inset = scored("georef", affine(30), verification=1.6, gcps=1)
    nodes = {
        "p9__1": make_node("p9__1", [strong, scored(UNPLACED, None, 0)], base="p9"),
        "p9__2": make_node("p9__2", [inset, scored(UNPLACED, None, 0)], base="p9"),
    }
    assert build_edges(nodes, {}) == []
    assignment = solve(nodes, [], {}, 0.0, (-74.0, LAT0))
    assert nodes["p9__2"].hypotheses[assignment["p9__2"]].source == "georef"


def test_solve_deterministic_under_permutation():
    def build(order):
        nodes = {}
        for stem in order:
            nodes[stem] = make_node(
                stem,
                [
                    scored("georef", affine(0), verification=1.5),
                    scored("snap:0", affine(200), verification=1.5),
                    scored(UNPLACED, None, 0),
                ],
            )
        return nodes

    a = solve(build(["p1", "p2", "p3"]), [], {}, 0.0, (-74.0, LAT0))
    b = solve(build(["p3", "p1", "p2"]), [], {}, 0.0, (-74.0, LAT0))
    assert a == b


def test_no_verification_falls_back_to_cached():
    h = Hypothesis(
        source="snap:0",
        affine=affine(0),
        scores={"cached": {"verification": 1.7}},
    )
    unary_energy(h, False, 600.0, None, None)
    assert h.scores["unverified"] is True
    assert h.unary < 0  # the cached 1.7 beats the 1.25 gate


def test_unary_rung_term_allows_second_families():
    # ON any integer rung (x1 or x2) is free — second scale families are
    # legitimate (fargo's 9-series). Only a between-rung scale pays.
    on_rung = Hypothesis(source="a", affine=affine(), scores={"verification": 1.5})
    double_rung = Hypothesis(
        source="b", affine=affine(scale=2.0), scores={"verification": 1.5}
    )
    between = Hypothesis(
        source="c", affine=affine(scale=1.45), scores={"verification": 1.5}
    )
    family = pose_scale_log2(affine())
    unary_energy(on_rung, False, 600.0, family, None)
    unary_energy(double_rung, False, 600.0, family, None)
    unary_energy(between, False, 600.0, family, None)
    assert on_rung.unary == double_rung.unary
    assert between.unary > on_rung.unary


def test_keep_bar_is_lower_than_enter_bar():
    # A modest real fit (ver 0.6) STAYS on a placed page but the same
    # evidence cannot ENTER an unplaced page — the pipeline's own keep/enter
    # asymmetry (INCUMBENT_DEFENSIBLE_VERIFICATION vs PRODUCTION_GATE_SCORE).
    kept = scored("georef", affine(0), verification=0.6, gcps=2, published=True)
    entrant = scored("snap:0", affine(0), verification=0.6, page_placed=False)
    strong_entrant = scored("snap:1", affine(0), verification=1.6, page_placed=False)
    unplaced = scored(UNPLACED, None, 0.0)
    assert kept.unary < unplaced.unary
    assert entrant.unary > unplaced.unary
    assert strong_entrant.unary < unplaced.unary


def test_publish_writes_sidecars_and_holds_unplaced(tmp_path):
    # --publish is the only mode that writes to the volume root, and it answers
    # for EVERY page: a chosen pose becomes pN.georef-final.json with corners,
    # a page arbitrated to unplaced becomes one with corners: null. The channel
    # sidecars are left exactly where they are -- unpublishing is a written
    # decision now, not the absence of a file.
    from mapsnap.reconcile import publish

    write_sidecar(tmp_path, "p1", "georef", georef_doc(affine(0)))
    write_sidecar(tmp_path, "p2", "georef-snap", georef_doc(affine(500)))
    keep = make_node("p1", [scored("georef", affine(0), verification=1.9, gcps=4)])
    drop = make_node(
        "p2",
        [
            scored("georef-snap", affine(500), verification=0.1),
            scored(UNPLACED, None, 0),
        ],
    )
    written, unplaced = publish(tmp_path, {"p1": keep, "p2": drop}, {"p1": 0, "p2": 1})
    assert (written, unplaced) == (1, 1)
    assert json.loads((tmp_path / "p1.georef-final.json").read_text())["corners"]
    assert (
        json.loads((tmp_path / "p2.georef-final.json").read_text())["corners"] is None
    )
    # p2's own channel sidecar is untouched: the pose stays readable.
    assert (tmp_path / "p2.georef-snap.json").exists()
    # expand_georef_globs skips the poseless one, so p2 goes unpublished.
    from mapsnap.make_iiif_georef import expand_georef_globs

    chosen = expand_georef_globs(f"{tmp_path}/p*.georef-final.json")
    assert [Path(p).name for p in chosen] == ["p1.georef-final.json"]


def test_publish_records_provenance(tmp_path):
    from mapsnap.reconcile import publish

    write_sidecar(tmp_path, "p1", "georef", georef_doc(affine(0)))
    node = make_node("p1", [scored("georef", affine(0), verification=1.9, gcps=4)])
    publish(tmp_path, {"p1": node}, {"p1": 0})
    doc = json.loads((tmp_path / "p1.georef-final.json").read_text())
    assert doc["reconcile"]["source"] == "georef"
    assert "terms" in doc["reconcile"] and "corners" in doc


def test_demoted_channel_is_not_the_incumbent(tmp_path):
    # A demotion is a verdict inside the file now, not a rename, so the sidecar
    # exists either way. The incumbent must follow the verdict: this is the
    # exact set the old glob-over-renamed-files arrangement published.
    from mapsnap import sidecar

    doc = georef_doc(affine(0))
    write_sidecar(tmp_path, "p1", "georef", doc)
    assert published_channel(tmp_path, "p1") == "georef"
    sidecar.demote(tmp_path / "p1.georef.json", sidecar.MISSCALE)
    assert published_channel(tmp_path, "p1") is None
    # ...but the pose is still a hypothesis, carrying its verdict.
    hypotheses, published = collect_hypotheses(tmp_path, "p1", None, None)
    assert published is None
    assert [h.source for h in hypotheses] == ["georef:misscale", UNPLACED]
    assert hypotheses[0].status == sidecar.MISSCALE


def test_rejected_poses_become_hypotheses(tmp_path):
    # The p55 shape under the collapsed layout: georef's key-map retry kept one
    # pose and set the other aside, both inside p55.georef.json. Both have to
    # reach the arbiter -- weighing them against each other is the point.
    from mapsnap import sidecar

    doc = georef_doc(affine(0))
    write_sidecar(tmp_path, "p55", "georef", doc)
    sidecar.attach_rejected(
        tmp_path / "p55.georef.json",
        [{**georef_doc(affine(5000)), "status": sidecar.KEYMAP_OUTLIER}],
    )
    hypotheses, published = collect_hypotheses(tmp_path, "p55", None, None)
    assert [h.source for h in hypotheses] == [
        "georef",
        "georef:keymap-outlier",
        UNPLACED,
    ]
    assert published == 0


def test_contradicted_term_follows_the_verdict_not_the_filename():
    # W_CONTRADICTED used to key off "contradicted" appearing in the sidecar's
    # NAME; it keys off the recorded status now.
    from mapsnap import sidecar
    from mapsnap.reconcile import W_CONTRADICTED

    plain = Hypothesis(source="georef", affine=affine(0), effective_gcps=2)
    plain.scores["verification"] = 1.0
    unary_energy(plain, False, 600.0, None, None)
    flagged = Hypothesis(
        source="georef",
        affine=affine(0),
        effective_gcps=2,
        status=sidecar.CONTRADICTED,
    )
    flagged.scores["verification"] = 1.0
    unary_energy(flagged, False, 600.0, None, None)
    assert flagged.unary - plain.unary == pytest.approx(W_CONTRADICTED)
