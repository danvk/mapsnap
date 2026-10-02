"""Place a sheet from another edition's fit of the same sheet (#34).

Sanborn re-issued a volume every decade or two with the same sheet numbering:
sheet 2 of Queens vol. 1 draws the same blocks in 1898, 1915, 1936, 1947 and
1950. Once one edition places a sheet, the others only need the pose between
the two scans of it, and that is a page-to-page match, not a search of the
city: the donor's road-UNet P(road) is warped into a local metre frame at its
fitted pose, and the target's P(road) is matched against it with edge-join's
masked NCC and chamfer refinement, over a small ladder of rotations and scales
around the donor's own. No street name is read, so a sheet whose labels OCR
could not read, or whose streets have since been renamed, still transfers.

Nothing here proves two editions share a numbering. A transfer is accepted on
how well the two road maps agree (inlier fraction and fine correlation), which
a renumbered sheet drawing different blocks fails.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap.compare_iiif_georef import skeleton_base
from mapsnap.edge_join import (
    FrameSpec,
    JoinCandidate,
    MatchParams,
    match_at_rotation,
    refine_and_rank,
    skeleton_points,
    warp_page,
)
from mapsnap.feature_index import FeatureIndex
from mapsnap.osm_snap import frame_around, osm_distance_m, osm_rasters, pose_theta_deg
from mapsnap.osm_to_centerlines import load_centerlines
from mapsnap.road_model import page_world_affine
from mapsnap.roadprob import load_roadprob

# Both scans draw the whole sheet, so near-total overlap is the expected
# answer, unlike edge-join where it is physically impossible.
TRANSFER_PARAMS = MatchParams(
    resolution_m=2.0,
    blur_sigma_m=6.0,
    fine_sigma_m=3.0,
    min_overlap_m2=20_000.0,
    max_overlap_frac=1.0,
    top_k=3,
    peak_separation_m=40.0,
)
# How far the target sheet's centre may sit from the donor's.
SEARCH_RADIUS_M = 250.0
# Ground beyond the donor sheet that the frame keeps, so a shifted target
# still overlaps it.
FRAME_MARGIN_M = 300.0
# The two scans of a sheet are the same paper at about the same resolution,
# but a re-drawn sheet can be re-cut and a scan can sit a few degrees askew.
SCALE_LADDER = (0.9, 0.95, 1.0, 1.05, 1.1)
ROTATION_LADDER_DEG = (-4.0, -2.0, 0.0, 2.0, 4.0)
# Acceptance: the share of the target's road skeleton within 5 m of the
# donor's, and the correlation of the two lightly-blurred road maps.
MIN_INLIER_FRAC = 0.5
MIN_NCC_FINE = 0.35
# Two scans of one sheet land nearly on top of each other. On a regular
# street grid a neighbouring sheet also matches the road map well, but only
# slid a block or more over onto the strip the two sheets share: on Queens
# vol. 1 true sheets shifted 1-14 m with >= 97% overlap, neighbours 87-236 m
# with <= 76%.
MAX_SHIFT_M = 75.0
MIN_OVERLAP_FRAC = 0.8
# A transfer and the target's own fit agree when their corners are this close.
AGREE_M = 30.0
# OSM agreement, logged beside each transfer and conflict: the share of the
# sheet's road skeleton within OSM_WITHIN_M of an OSM centerline. A
# diagnostic only (see decide).
OSM_WITHIN_M = 10.0
# Two editions' fits of a sheet vote for the same pose when their centres are
# this close. Scans of one sheet place within 4-6 m of each other; a sheet
# redrawn between editions moves ~100 m.
VOTE_RADIUS_M = 50.0


@dataclass
class Sheet:
    """One edition's scan of a sheet: its road map, and its fit if it has one."""

    key: str
    prob: np.ndarray
    georef: dict | None


@dataclass
class Transfer:
    """A target sheet placed from a donor edition's fit of the same sheet."""

    key: str
    donor: str
    donor_key: str  # the donor's sheet: the same number unless a scan is mislabeled
    corners: list[list[float]]
    width: int
    height: int
    ncc: float
    ncc_fine: float
    inlier_frac: float
    chamfer_mean_m: float
    scale_adjust: float
    rotation_deg: float
    scale_ratio: float
    shift_m: float  # target centre from the donor's, on the ground
    overlap_frac: float  # of the target's footprint inside the donor's

    @property
    def accepted(self) -> bool:
        return (
            self.inlier_frac >= MIN_INLIER_FRAC
            and self.ncc_fine >= MIN_NCC_FINE
            and self.shift_m <= MAX_SHIFT_M
            and self.overlap_frac >= MIN_OVERLAP_FRAC
        )

    def georef(self) -> dict:
        """The transfer as a georef-final sidecar."""
        return {
            "width": self.width,
            "height": self.height,
            "corners": self.corners,
            "edition_transfer": asdict(self),
        }


def read_georef(run_dir: Path, key: str) -> dict | None:
    """A page's published pose from a run's georef-final sidecar, or None if unplaced."""
    path = run_dir / f"{key}.georef-final.json"
    if not path.exists():
        return None
    georef = json.loads(path.read_text())
    return georef if georef.get("corners") else None


def sheet_keys(volume: Path) -> list[str]:
    """The unsplit map sheets of a volume with a road map: p1, p2, ...

    Not key maps, panels, or a skeleton (pNs) beside its full-color sheet,
    which publication drops anyway.
    """
    keys = {
        path.name.removesuffix(".roadprob.jpg")
        for path in volume.glob("p*.roadprob.jpg")
    }
    return sorted(
        (
            k
            for k in keys
            if "__" not in k and k != "p0" and skeleton_base(k, keys) is None
        ),
        key=lambda k: (int("".join(c for c in k if c.isdigit()) or 0), k),
    )


def load_sheet(volume: Path, key: str, georef: dict | None) -> Sheet | None:
    """A sheet's road map and pose; None when it has no road map."""
    prob = load_roadprob(volume / f"{key}.jpg")
    if prob is None:
        return None
    return Sheet(key=key, prob=prob.astype(np.float32), georef=georef)


def corner_distance_m(a: list, b: list) -> float:
    """Mean distance between two corner quads, in metres."""
    a_xy, b_xy = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    kx = 111_320.0 * math.cos(math.radians(float(a_xy[:, 1].mean())))
    delta = (a_xy - b_xy) * (kx, 110_540.0)
    return float(np.linalg.norm(delta, axis=1).mean())


def donor_frame(
    donor: Sheet, params: MatchParams
) -> tuple[FrameSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The donor sheet in a local metre frame: (frame, pose, prob, valid, distance_m).

    ``pose`` maps donor page px to raster px; ``distance_m`` is the clamped
    distance to the donor's own road skeleton, for chamfer refinement.
    """
    assert donor.georef is not None
    affine = page_world_affine(donor.georef)
    height, width = donor.prob.shape
    center = affine @ np.array([width / 2, height / 2, 1.0])
    origin = (float(center[0]), float(center[1]))
    kx = 111_320.0 * math.cos(math.radians(origin[1]))
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    metres = (corners @ affine.T - center) * (kx, 110_540.0)
    x_min = float(metres[:, 0].min()) - FRAME_MARGIN_M
    x_max = float(metres[:, 0].max()) + FRAME_MARGIN_M
    y_min = float(metres[:, 1].min()) - FRAME_MARGIN_M
    y_max = float(metres[:, 1].max()) + FRAME_MARGIN_M
    res = params.resolution_m
    shape = (
        math.ceil((y_max - y_min) / res),
        math.ceil((x_max - x_min) / res),
    )
    frame = FrameSpec(origin=origin, x_min=x_min, y_max=y_max, res_m=res, shape=shape)
    pose = frame.page_to_raster_affine(affine)
    prob, valid = warp_page(donor.prob, pose, shape)
    skeleton = np.zeros(shape, dtype=bool)
    points = skeleton_points(donor.prob, params.mask_threshold, params.mask_min_area)
    if len(points):
        placed = np.column_stack([points, np.ones(len(points))]) @ pose.T
        cols = placed[:, 0].round().astype(int)
        rows = placed[:, 1].round().astype(int)
        inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        skeleton[rows[inside], cols[inside]] = True
    return frame, pose, prob, valid, osm_distance_m(skeleton, res)


def transfer_sheet(
    target: Sheet, donor: Sheet, donor_name: str, params: MatchParams = TRANSFER_PARAMS
) -> Transfer | None:
    """Place ``target`` by matching its road map to ``donor``'s at the donor's pose.

    None when the donor has no fit or no candidate survives refinement.
    """
    if donor.georef is None:
        return None
    frame, pose, fixed, valid, distance_m = donor_frame(donor, params)
    donor_h, donor_w = donor.prob.shape
    target_h, target_w = target.prob.shape
    donor_theta = pose_theta_deg(pose)
    donor_scale = math.sqrt(abs(np.linalg.det(pose[:, :2])))
    # The two scans draw the same paper, so the target's raster scale is the
    # donor's times the ratio of their pixel widths.
    base_scale = donor_scale * donor_w / target_w
    center = pose @ np.array([donor_w / 2, donor_h / 2, 1.0])
    sigma_px = max(params.blur_sigma_m / params.resolution_m, 0.5)
    fixed_blur = cv2.GaussianBlur(fixed * valid, (0, 0), sigma_px)
    candidates: list[JoinCandidate] = []
    for ratio in SCALE_LADDER:
        for offset in ROTATION_LADDER_DEG:
            for candidate in match_at_rotation(
                fixed_blur,
                valid,
                target.prob,
                scale=base_scale * ratio,
                theta=donor_theta + offset,
                params=params,
                search_center=(float(center[0]), float(center[1])),
                search_radius_px=SEARCH_RADIUS_M / params.resolution_m,
            ):
                candidate.scale = ratio  # type: ignore[attr-defined]
                candidates.append(candidate)
    if not candidates:
        return None
    # Keep refinement affordable: the best few NCC peaks across the ladder.
    candidates = sorted(candidates, key=lambda c: -c.ncc)[:12]
    points = skeleton_points(target.prob, params.mask_threshold, params.mask_min_area)
    region = cv2.dilate(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ranked = refine_and_rank(
        candidates,
        distance_m,
        points,
        fixed_valid=valid,
        page_shape=target.prob.shape,
        max_overlap_frac=1.0,
        region=region,
        fixed_prob=fixed,
        target_prob=target.prob,
        fine_sigma_px=params.fine_sigma_m / params.resolution_m,
        solve_scale=True,
    )
    best = ranked[0]
    if not math.isfinite(best.verification_score()):
        return None
    world = frame.raster_pose_to_world_affine(best.pose)
    quad = np.array(
        [[0, 0, 1], [target_w, 0, 1], [target_w, target_h, 1], [0, target_h, 1]]
    )
    corners = (quad @ world.T).tolist()
    best_scale = math.sqrt(abs(np.linalg.det(best.pose[:, :2])))
    return Transfer(
        key=target.key,
        donor=donor_name,
        donor_key=donor.key,
        corners=corners,
        width=target_w,
        height=target_h,
        ncc=round(best.ncc, 4),
        ncc_fine=round(best.ncc_fine, 4),
        inlier_frac=round(best.inlier_frac, 4),
        chamfer_mean_m=round(best.chamfer_mean_m, 2),
        scale_adjust=round(best.scale_adjust, 4),
        rotation_deg=round(pose_theta_deg(best.pose) - donor_theta, 2),
        scale_ratio=round(best_scale / base_scale, 4),
        shift_m=round(
            float(np.linalg.norm(best.pose @ [target_w / 2, target_h / 2, 1] - center))
            * params.resolution_m,
            1,
        ),
        overlap_frac=round(best.overlap_frac, 3),
    )


@dataclass
class Edition:
    """One edition of the volume: its directory, year, and per-sheet poses."""

    volume: Path
    year: int
    corpus: dict[str, dict]  # key -> georef from the corpus run
    final: dict[str, dict]  # key -> georef after transfers (filled in as we go)


def load_edition(volume: Path, corpus_run: str) -> Edition:
    """An edition's corpus poses, from its run's georef-final sidecars."""
    metadata = json.loads((volume / "metadata.json").read_text())
    run_dir = volume / "runs" / corpus_run
    corpus = {
        key: georef
        for key in sheet_keys(volume)
        if (georef := read_georef(run_dir, key)) is not None
    }
    return Edition(volume=volume, year=int(metadata["year"]), corpus=corpus, final={})


def donors_for(target: Edition, editions: list[Edition]) -> list[Edition]:
    """The other editions in the order to try them: newer ones nearest first, then older.

    Fits flow from the newest edition back. An older edition's fit is only a
    fallback: a near-perfect image match copies a donor's error as faithfully
    as its pose, and Queens 1898 p70's 1.9 km disaster, transferred to 1915
    as its nearest edition, cleared every gate (grid aliasing fooled OSM too).
    """
    others = [e for e in editions if e is not target]
    return sorted(
        others, key=lambda e: (e.year < target.year, abs(e.year - target.year))
    )


def sheet_number(key: str) -> tuple[int, str] | None:
    """A page key's sheet number and suffix: p77 -> (77, ""), p0005N -> (5, "N")."""
    match = re.fullmatch(r"p0*(\d+)([A-Za-z]*)", key)
    return (int(match[1]), match[2]) if match else None


def same_sheet(a: str, b: str) -> bool:
    """Whether two editions' page keys name the same sheet.

    The same number, with the same suffix or with none on one side: Chicago
    vol. 1 is p5 in 1906 but p5N in 1950, its North part. Lettered subsheets
    (p10Sa beside p10Sb) are only ever tried; the gates decide.
    """
    a_sheet, b_sheet = sheet_number(a), sheet_number(b)
    if a_sheet is None or b_sheet is None:
        return a == b
    return a_sheet[0] == b_sheet[0] and (
        a_sheet[1] == b_sheet[1] or not a_sheet[1] or not b_sheet[1]
    )


def edition_keys(edition: Edition, key: str) -> list[str]:
    """The keys an edition places for the sheet ``key`` names: itself, else its namesakes."""
    placed = edition.final.keys() | edition.corpus.keys()
    if key in placed:
        return [key]
    return sorted(k for k in placed if same_sheet(key, k))


def neighbour_keys(key: str) -> list[str]:
    """The sheet numbers either side of a key, same suffix: p77 -> [p76, p78]."""
    sheet = sheet_number(key)
    if sheet is None:
        return []
    number, suffix = sheet
    return [f"p{n}{suffix}" for n in (number - 1, number + 1) if n > 0]


def find_transfer(
    sheet: Sheet, target: Edition, editions: list[Edition], wanted: list[str]
) -> tuple[Transfer | None, list[str]]:
    """The first accepted transfer from a donor sheet named by ``wanted``, in donor order.

    Each wanted key is resolved in each donor edition (see edition_keys).
    Returns the transfer with the "year:key" donors tried; when none is
    accepted, the last attempt at the sheet's own number (for the log) or None.
    """
    fallback = None
    tried: list[str] = []
    for wanted_key in wanted:
        for donor_edition in donors_for(target, editions):
            for donor_key in edition_keys(donor_edition, wanted_key):
                georef = donor_edition.final.get(donor_key) or (
                    donor_edition.corpus.get(donor_key)
                )
                if georef is None:
                    continue
                donor = load_sheet(donor_edition.volume, donor_key, georef)
                if donor is None:
                    continue
                transfer = transfer_sheet(sheet, donor, donor_edition.volume.name)
                tried.append(f"{donor_edition.year}:{donor_key}")
                if transfer is None:
                    continue
                if transfer.accepted:
                    return transfer, tried
                if wanted_key == sheet.key:
                    fallback = transfer
    return fallback, tried


def osm_agreement(
    prob: np.ndarray,
    georef: dict,
    features: FeatureIndex,
    params: MatchParams = TRANSFER_PARAMS,
) -> float:
    """Share of a sheet's road skeleton within OSM_WITHIN_M of an OSM centerline at a pose.

    A neutral referee between two poses for one sheet: it reads no names and
    trusts neither edition's fit.
    """
    affine = page_world_affine(georef)
    height, width = prob.shape
    center = affine @ np.array([width / 2, height / 2, 1.0])
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    kx = 111_320.0 * math.cos(math.radians(float(center[1])))
    half_m = float(np.abs((corners @ affine.T - center) * (kx, 110_540.0)).max())
    frame = frame_around(
        (float(center[0]), float(center[1])),
        half_m=half_m + FRAME_MARGIN_M / 6,
        res_m=params.resolution_m,
    )
    _, _, skeleton = osm_rasters(frame, features)
    distance = osm_distance_m(skeleton, params.resolution_m)
    points = skeleton_points(prob, params.mask_threshold, params.mask_min_area)
    if not len(points):
        return 0.0
    placed = np.column_stack([points, np.ones(len(points))]) @ (
        frame.page_to_raster_affine(affine).T
    )
    rows = np.clip(placed[:, 1].round().astype(int), 0, frame.shape[0] - 1)
    cols = np.clip(placed[:, 0].round().astype(int), 0, frame.shape[1] - 1)
    return float((distance[rows, cols] <= OSM_WITHIN_M).mean())


def center_distance_m(a: list, b: list) -> float:
    """Distance between the centres of two corner quads, in metres."""
    a_center = np.asarray(a, dtype=float).mean(axis=0)
    b_center = np.asarray(b, dtype=float).mean(axis=0)
    kx = 111_320.0 * math.cos(math.radians(float(a_center[1])))
    return float(np.hypot(*((a_center - b_center) * (kx, 110_540.0))))


def edition_votes(key: str, corners: list, voters: list[Edition]) -> list[int]:
    """The ``voters`` whose own (corpus) fit of ``key`` lands where ``corners`` does.

    Each edition was fitted independently, so agreement between them is
    evidence that neither OSM nor any one edition can supply on its own.
    """
    return [
        edition.year
        for edition in voters
        if any(
            (georef := edition.corpus.get(other)) is not None
            and center_distance_m(georef["corners"], corners) <= VOTE_RADIUS_M
            for other in edition_keys(edition, key)
        )
    ]


def decide(
    own_placed: bool,
    transfer: Transfer | None,
    disagreement_m: float | None = None,
    votes: tuple[int, int] | None = None,
) -> str:
    """What a sheet publishes: "own", "transfer", "replaced" or "unplaced".

    When an accepted transfer disagrees with the sheet's own fit by more
    than AGREE_M, one of the two editions' fits is wrong. ``votes`` is
    (own, transfer): how many other editions' independent fits, not the
    donor's, back each pose. The transfer replaces the own fit only when it
    has more; a tie keeps the own fit. OSM agreement used to break ties, and
    on Chicago it picked the wrong fit four times in six: on a street grid a
    pose 100-200 m off agrees with OSM about as well as the right one.
    """
    accepted = transfer is not None and transfer.accepted
    if not own_placed:
        return "transfer" if accepted else "unplaced"
    if not accepted or disagreement_m is None or disagreement_m <= AGREE_M:
        return "own"
    if votes is not None and votes[1] > votes[0]:
        return "replaced"
    return "own"


def transfer_edition(
    target: Edition, editions: list[Edition], features: FeatureIndex, log: list[dict]
) -> dict[str, dict]:
    """Final poses for every sheet of ``target``: its own fit, or a transfer.

    Each sheet is matched against the nearest edition that places it (that
    edition's final pose when already settled, its corpus pose otherwise).
    See ``decide`` for which pose a sheet keeps.
    """
    final: dict[str, dict] = {}
    for key in sheet_keys(target.volume):
        own = target.corpus.get(key)
        sheet = load_sheet(target.volume, key, own)
        if sheet is None:
            continue
        transfer, tried = find_transfer(sheet, target, editions, [key])
        if own is None and not (transfer is not None and transfer.accepted):
            # A mislabeled scan (#555: Queens 1898's p76-p80 are printed
            # sheets 77-81) matches the donor's sheet one number over.
            relabeled, more = find_transfer(
                sheet, target, editions, neighbour_keys(key)
            )
            tried += more
            if relabeled is not None and relabeled.accepted:
                transfer = relabeled
        entry: dict = {"edition": target.year, "key": key, "own": own is not None}
        disagreement = None
        osm = None
        votes = None
        if transfer is not None:
            entry |= asdict(transfer) | {"accepted": transfer.accepted, "tried": tried}
            if transfer.accepted:
                transfer_osm = osm_agreement(sheet.prob, transfer.georef(), features)
                entry["transfer_osm"] = round(transfer_osm, 3)
            if own is not None:
                disagreement = corner_distance_m(own["corners"], transfer.corners)
                entry["own_vs_transfer_m"] = round(disagreement, 1)
                if transfer.accepted and disagreement > AGREE_M:
                    osm = (
                        osm_agreement(sheet.prob, own, features),
                        entry["transfer_osm"],
                    )
                    entry["own_osm"] = round(osm[0], 3)
                    # Neither the target nor the donor is a witness: a
                    # transfer lands on its donor's fit by construction, so
                    # with two editions the donor would always outvote.
                    voters = [
                        e
                        for e in editions
                        if e is not target and e.volume.name != transfer.donor
                    ]
                    own_votes = edition_votes(key, own["corners"], voters)
                    transfer_votes = edition_votes(key, transfer.corners, voters)
                    votes = (len(own_votes), len(transfer_votes))
                    entry["own_votes"] = own_votes
                    entry["transfer_votes"] = transfer_votes
        status = decide(own is not None, transfer, disagreement, votes)
        entry["status"] = status
        log.append(entry)
        if status == "own":
            assert own is not None
            final[key] = own
        elif status in ("transfer", "replaced"):
            assert transfer is not None
            final[key] = transfer.georef()
        detail = (
            f"donor {transfer.donor} {transfer.donor_key} inl {transfer.inlier_frac:.2f} "
            f"ncc {transfer.ncc_fine:.2f} rot {transfer.rotation_deg:+.1f} "
            f"scale {transfer.scale_ratio:.3f} shift {transfer.shift_m:.0f} m "
            f"overlap {transfer.overlap_frac:.2f}"
            + (f" osm {entry['transfer_osm']:.2f}" if "transfer_osm" in entry else "")
            + (f" vs own {disagreement:.0f} m" if disagreement is not None else "")
            + (f" (own osm {osm[0]:.2f})" if osm is not None else "")
            + (f" votes {votes[0]}-{votes[1]}" if votes is not None else "")
            if transfer is not None
            else "no donor"
        )
        print(f"  {target.year} {key:5s} {status:8s} {detail}", flush=True)
    return final


def corpus_panels(edition: Edition, corpus_run: str) -> dict[str, Path]:
    """Sidecars of the corpus run's split sheets that no transfer replaces.

    Transfers are whole sheets, so a split sheet the edition has no transfer
    for keeps its corpus panels: each placed ``pN__i`` georef and the parent's
    ``pN.panels.json``, which publication needs to place a panel on its canvas.
    """
    run_dir = edition.volume / "runs" / corpus_run
    files: dict[str, Path] = {}
    for panels in run_dir.glob("*.panels.json"):
        parent = panels.name.removesuffix(".panels.json")
        if parent in edition.final:
            continue
        placed = [
            path
            for path in run_dir.glob(f"{parent}__*.georef-final.json")
            if json.loads(path.read_text()).get("corners")
        ]
        if placed:
            files[panels.name] = panels
            files |= {path.name: path for path in placed}
    return files


def write_run(edition: Edition, run: str, work: Path, corpus_run: str) -> Path:
    """Build the edition's annotation page from its final poses with ``mapsnap iiif``.

    The pipeline publishes from a directory holding each page image beside its
    sidecar, so this lays one out under ``work`` (images symlinked) and copies
    the result to ``<volume>/runs/<run>/mapsnap.iiif.json``.
    """
    work.mkdir(parents=True, exist_ok=True)
    for path in edition.volume.iterdir():
        if path.suffix == ".jpg" or path.name in (
            "metadata.json",
            "centerlines.osm.pbf",
        ):
            link = work / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
    for stale in [*work.glob("*.georef-final.json"), *work.glob("*.panels.json")]:
        stale.unlink()
    for key, georef in edition.final.items():
        (work / f"{key}.georef-final.json").write_text(json.dumps(georef, indent=1))
    panels = corpus_panels(edition, corpus_run)
    for name, path in panels.items():
        shutil.copy(path, work / name)
    output = work / "mapsnap.iiif.json"
    subprocess.run(
        [
            "mapsnap",
            "iiif",
            str(work / "metadata.json"),
            str(work / "*.georef-final.json"),
            "--centerlines",
            str(work / "centerlines.osm.pbf"),
            "--output",
            str(output),
            "--run-tag",
            run,
        ],
        check=True,
    )
    run_dir = edition.volume / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(output, run_dir / "mapsnap.iiif.json")
    for key, georef in edition.final.items():
        (run_dir / f"{key}.georef-final.json").write_text(json.dumps(georef, indent=1))
    for name, path in panels.items():
        shutil.copy(path, run_dir / name)
    return run_dir / "mapsnap.iiif.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Place each edition's unplaced sheets from other editions' fits of the "
            "same sheet (#34). Editions are processed newest first, so an older "
            "edition can inherit a sheet an intermediate edition only gained by "
            "transfer."
        )
    )
    parser.add_argument("volumes", nargs="+", type=Path, help="Edition directories")
    parser.add_argument(
        "--corpus-run", default="corpus-v1", help="Run holding the fits"
    )
    parser.add_argument("--run", default="edition-transfer", help="Output run tag")
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Scratch directory for laying out each edition's publication",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Republish an existing run's sheet sidecars without matching again",
    )
    args = parser.parse_args()

    editions = sorted(
        (load_edition(v, args.corpus_run) for v in args.volumes), key=lambda e: -e.year
    )
    if args.rebuild:
        for edition in editions:
            run_dir = edition.volume / "runs" / args.run
            edition.final = {
                key: georef
                for key in sheet_keys(edition.volume)
                if (georef := read_georef(run_dir, key)) is not None
            }
            path = write_run(
                edition, args.run, args.work / edition.volume.name, args.corpus_run
            )
            print(f"{path}: {len(edition.final)} sheets", file=sys.stderr)
        return
    log: list[dict] = []
    features_by_path: dict[Path, FeatureIndex] = {}
    for edition in editions:
        print(f"{edition.volume.name}: {len(edition.corpus)} sheets placed", flush=True)
        centerlines = (edition.volume / "centerlines.osm.pbf").resolve()
        if centerlines not in features_by_path:
            collection = load_centerlines(centerlines)
            features_by_path[centerlines] = FeatureIndex(collection["features"])
        edition.final = transfer_edition(
            edition, editions, features_by_path[centerlines], log
        )
    for edition in editions:
        path = write_run(
            edition, args.run, args.work / edition.volume.name, args.corpus_run
        )
        gained = len(edition.final) - len(edition.corpus)
        print(f"{path}: {len(edition.final)} sheets ({gained:+d})", file=sys.stderr)
    for edition in editions:
        with (edition.volume / "runs" / args.run / "transfers.jsonl").open("w") as out:
            for entry in log:
                if entry["edition"] == edition.year:
                    out.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
