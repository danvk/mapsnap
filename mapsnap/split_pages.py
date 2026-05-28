"""Detect and split Sanborn map pages that contain multiple maps.

Some Sanborn pages contain several independent maps arranged side-by-side,
separated by thick black border lines (flanked by thinner lines). This script
detects those dividing lines and writes each map section as a separate JPEG.

The detection works by downsampling the image to 1/DOWNSAMPLE resolution, then
looking for rows or columns that contain a long consecutive run of dark pixels.
A run spanning at least MIN_RUN_FRACTION of the relevant dimension is treated as
evidence of a split line. Contiguous bands of such rows/columns that are at least
MIN_THICKNESS_PX wide are identified as split lines. Bands too close to the image
edge (page border frames) are ignored via EDGE_MARGIN_FRACTION.

The image is partitioned recursively: vertical splits first (producing left-to-right
columns), then horizontal splits within each column (producing top-to-bottom rows).

Diagonal splits are not yet supported.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import scale as shapely_scale
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box

from mapsnap.utils import image_stem

# Work at 1/DOWNSAMPLE of original resolution for speed.
DOWNSAMPLE = 4
# Pixels below this grayscale value are treated as "dark". Sanborn maps are color
# JPEGs, so even "black" lines rarely have values below 50 after JPEG compression;
# 128 comfortably captures them.
DARK_THRESHOLD = 128
# A split line must produce a consecutive dark run spanning at least this fraction
# of the relevant image dimension (height for vertical lines, width for horizontal).
MIN_RUN_FRACTION = 0.50
# A band of qualifying rows/columns must be at least this many pixels wide in
# downsampled coordinates to be counted as a split line (~12px in the original).
MIN_THICKNESS_PX = 3
# Bands whose center falls within this fraction of the image edge are treated as
# page border frames and ignored.
EDGE_MARGIN_FRACTION = 0.05
# Secondary-pass parameters for corner-inset detection (used when primary pass finds no split).
# Lower run threshold to catch insets that span ~20-50% of the image dimension.
INSET_MIN_RUN_FRACTION = 0.15
# Upper bound on band width (downsampled coords) to reject large dark urban areas that
# are not split lines. Detroit false-positive bands have width 23-62; real inset borders
# have width 4-7, so 15 is a safe upper bound.
INSET_MAX_THICKNESS_PX = 15


def _max_run(arr: np.ndarray) -> int:
    """Length of the longest consecutive run of True values in a 1-D boolean array."""
    if not arr.any():
        return 0
    padded = np.concatenate([[False], arr, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())


def find_thick_bands(
    profile: np.ndarray, min_run: int, min_thickness: int
) -> list[tuple[int, int]]:
    """Find contiguous bands where the longest-run profile exceeds a threshold.

    Args:
        profile: 1-D array where profile[i] is the longest consecutive run of dark
                 pixels in row i (for horizontal detection) or column i (for vertical).
        min_run: A row/column qualifies when its profile value exceeds this.
        min_thickness: Minimum number of consecutive qualifying indices to form a band.

    Returns:
        List of (start, end) inclusive index pairs for each qualifying band.
    """
    qualifying = profile > min_run
    bands: list[tuple[int, int]] = []
    in_band = False
    band_start = 0
    n = len(qualifying)
    for i in range(n):
        if qualifying[i] and not in_band:
            in_band = True
            band_start = i
        elif not qualifying[i] and in_band:
            in_band = False
            if i - band_start >= min_thickness:
                bands.append((band_start, i - 1))
    if in_band and n - band_start >= min_thickness:
        bands.append((band_start, n - 1))
    return bands


def _col_max_runs(dark: np.ndarray) -> np.ndarray:
    """Per-column longest vertical run of dark pixels."""
    n_cols = dark.shape[1]
    profile = np.zeros(n_cols, dtype=np.int32)
    for j in range(n_cols):
        profile[j] = _max_run(dark[:, j])
    return profile


def _row_max_runs(dark: np.ndarray) -> np.ndarray:
    """Per-row longest horizontal run of dark pixels."""
    n_rows = dark.shape[0]
    profile = np.zeros(n_rows, dtype=np.int32)
    for i in range(n_rows):
        profile[i] = _max_run(dark[i, :])
    return profile


def _trace_vertical_band_edges(
    dark: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row left and right extents of dark pixels within dark[y0:y1, x0:x1].

    For each row, finds the leftmost and rightmost dark pixel inside the band.
    Gaps (rows with no dark pixels, e.g. a scale-bar cutout) are filled by linear
    interpolation so the boundary paths remain continuous.

    Returns (left_edge, right_edge):
      left_edge[i]  = absolute x of the leftmost dark pixel at row y0+i
                      (the right boundary of the section to the LEFT of the band)
      right_edge[i] = absolute x of (rightmost dark pixel + 1) at row y0+i
                      (the left boundary of the section to the RIGHT of the band)
    Both arrays have shape (y1-y0,) and dtype int32.
    """
    n = y1 - y0
    left_edge: np.ndarray = np.full(n, np.nan)
    right_edge: np.ndarray = np.full(n, np.nan)
    for i in range(n):
        xs = np.where(dark[y0 + i, x0:x1])[0]
        if xs.size:
            left_edge[i] = float(xs[0]) + x0
            right_edge[i] = float(xs[-1]) + x0 + 1.0
    for path in (left_edge, right_edge):
        known = np.where(~np.isnan(path))[0]
        if known.size > 0:
            path[:] = np.interp(np.arange(n), known, path[known])
        else:
            path[:] = (x0 + x1) / 2.0
    return left_edge.astype(np.int32), right_edge.astype(np.int32)


def _trace_horizontal_band_edges(
    dark: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column top and bottom extents of dark pixels within dark[y0:y1, x0:x1].

    Analogous to _trace_vertical_band_edges but for horizontal bands.

    Returns (top_edge, bot_edge):
      top_edge[j]  = absolute y of the topmost dark pixel at column x0+j
      bot_edge[j]  = absolute y of (bottommost dark pixel + 1) at column x0+j
    Both arrays have shape (x1-x0,) and dtype int32.
    """
    n = x1 - x0
    top_edge: np.ndarray = np.full(n, np.nan)
    bot_edge: np.ndarray = np.full(n, np.nan)
    for j in range(n):
        ys = np.where(dark[y0:y1, x0 + j])[0]
        if ys.size:
            top_edge[j] = float(ys[0]) + y0
            bot_edge[j] = float(ys[-1]) + y0 + 1.0
    for path in (top_edge, bot_edge):
        known = np.where(~np.isnan(path))[0]
        if known.size > 0:
            path[:] = np.interp(np.arange(n), known, path[known])
        else:
            path[:] = (y0 + y1) / 2.0
    return top_edge.astype(np.int32), bot_edge.astype(np.int32)


def _vertical_strip_polygon(
    left_path: np.ndarray, right_path: np.ndarray, y0: int, y1: int
) -> Polygon:
    """Polygon for a vertical strip bounded by two per-row x-position arrays.

    left_path and right_path have length y1-y0 (one value per row). The polygon
    extends from y=y0 to y=y1 by repeating the last path value at the boundary.
    """
    left_ext = np.append(left_path, left_path[-1])
    right_ext = np.append(right_path, right_path[-1])
    ys = list(range(y0, y1 + 1))
    pts = list(zip(left_ext.tolist(), ys)) + list(
        zip(right_ext[::-1].tolist(), ys[::-1])
    )
    return Polygon(pts).simplify(0.5, preserve_topology=True)  # type: ignore[return-value]


def _horizontal_strip_polygon(
    top_path: np.ndarray, bot_path: np.ndarray, x0: int, x1: int
) -> Polygon:
    """Polygon for a horizontal strip bounded by two per-column y-position arrays.

    top_path and bot_path have length x1-x0 (one value per column). The polygon
    extends from x=x0 to x=x1 by repeating the last path value at the boundary.
    """
    top_ext = np.append(top_path, top_path[-1])
    bot_ext = np.append(bot_path, bot_path[-1])
    xs = list(range(x0, x1 + 1))
    pts = list(zip(xs, top_ext.tolist())) + list(zip(xs[::-1], bot_ext[::-1].tolist()))
    return Polygon(pts).simplify(0.5, preserve_topology=True)  # type: ignore[return-value]


def _safe_intersection(region: Polygon, strip: Polygon) -> Polygon | None:
    """Intersect region with strip; return None if result is empty or degenerate."""
    result = region.intersection(strip)
    if result.is_empty:
        return None
    if isinstance(result, MultiPolygon):
        result = max(result.geoms, key=lambda g: g.area)
    if not isinstance(result, Polygon) or result.area < 1:
        return None
    return result


def _find_corner_inset(
    dark: np.ndarray,
) -> list[tuple[int, int, int, int]] | None:
    """Detect a corner-inset map using relaxed thresholds.

    Looks for a single vertical band AND a single horizontal band that together
    define a rectangular inset in one corner of the image. Returns [full_rect,
    inset_rect] ordered so the full image comes first (matching OIM convention),
    or None if no corner inset is found.

    The inset corner is determined by counting dark pixels in the two halves of
    the image: the half containing more dark content is where the inset sits.
    """
    h, w = dark.shape
    em_cols = int(EDGE_MARGIN_FRACTION * w)
    em_rows = int(EDGE_MARGIN_FRACTION * h)
    min_vrun = int(INSET_MIN_RUN_FRACTION * h)
    min_hrun = int(INSET_MIN_RUN_FRACTION * w)

    col_profile = _col_max_runs(dark)
    v_bands = find_thick_bands(col_profile, min_vrun, MIN_THICKNESS_PX)
    v_bands = [
        (s, e)
        for s, e in v_bands
        if em_cols < (s + e) // 2 < w - em_cols
        and (e - s + 1) <= INSET_MAX_THICKNESS_PX
    ]

    row_profile = _row_max_runs(dark)
    h_bands = find_thick_bands(row_profile, min_hrun, MIN_THICKNESS_PX)
    h_bands = [
        (s, e)
        for s, e in h_bands
        if em_rows < (s + e) // 2 < h - em_rows
        and (e - s + 1) <= INSET_MAX_THICKNESS_PX
    ]

    if len(v_bands) != 1 or len(h_bands) != 1:
        return None

    vb_start, vb_end = v_bands[0]
    hb_start, hb_end = h_bands[0]

    # Determine which corner the inset occupies by comparing dark-pixel density
    # in each half of the image.
    left_dark = int(dark[:, :vb_start].sum())
    right_dark = int(dark[:, vb_end + 1 :].sum())
    top_dark = int(dark[:hb_start, :].sum())
    bottom_dark = int(dark[hb_end + 1 :, :].sum())

    inset_left = left_dark > right_dark  # inset is on the left side
    inset_top = top_dark > bottom_dark  # inset is on the top side

    if inset_left and inset_top:
        inset = (0, 0, vb_start, hb_start)
    elif not inset_left and inset_top:
        inset = (vb_end + 1, 0, w, hb_start)
    elif inset_left and not inset_top:
        inset = (0, hb_end + 1, vb_start, h)
    else:
        inset = (vb_end + 1, hb_end + 1, w, h)

    return [(0, 0, w, h), inset]


def partition_image(
    img_ds: np.ndarray,
    min_run_fraction: float = MIN_RUN_FRACTION,
    min_thickness: int = MIN_THICKNESS_PX,
    edge_margin_fraction: float = EDGE_MARGIN_FRACTION,
) -> list[Polygon]:
    """Recursively partition a downsampled grayscale image into map sections.

    Detects thick black dividing lines, traces each line's per-row/per-column
    path to handle jogging splits, and builds Shapely polygon boundaries.
    Vertical splits are processed first (left-to-right), then horizontal splits
    within each resulting column (top-to-bottom).

    Args:
        img_ds: Downsampled grayscale image as a 2-D uint8 array (H, W).
        min_run_fraction: A row/column qualifies when its longest dark run exceeds
                          this fraction of the relevant dimension.
        min_thickness: Minimum band width in pixels (downsampled coordinates).
        edge_margin_fraction: Bands whose center is within this fraction of the image
                              edge are ignored (they are page border frames).

    Returns:
        List of Shapely Polygons in downsampled image coordinates,
        ordered left-to-right then top-to-bottom.
    """
    dark = img_ds < DARK_THRESHOLD
    h, w = dark.shape
    region = shapely_box(0, 0, w, h)
    sections = _partition_polygon(
        dark, region, min_run_fraction, min_thickness, edge_margin_fraction, depth=0
    )
    if len(sections) <= 1:
        corner = _find_corner_inset(dark)
        if corner is not None:
            return [shapely_box(cx0, cy0, cx1, cy1) for cx0, cy0, cx1, cy1 in corner]
    return sections


def _partition_polygon(
    dark: np.ndarray,
    region: Polygon,
    min_run_fraction: float,
    min_thickness: int,
    edge_margin_fraction: float,
    depth: int,
) -> list[Polygon]:
    """Recursively split region along thick dark bands, tracing each band's exact path."""
    if depth >= 3 or region.is_empty:
        return [region]

    x0, y0, x1, y1 = (int(v) for v in region.bounds)
    sub_h, sub_w = y1 - y0, x1 - x0
    if sub_h <= 0 or sub_w <= 0:
        return [region]

    sub = dark[y0:y1, x0:x1]

    # --- Vertical split lines ---
    min_vrun = int(min_run_fraction * sub_h)
    col_profile = _col_max_runs(sub)
    v_bands = find_thick_bands(col_profile, min_vrun, min_thickness)
    em_cols = int(edge_margin_fraction * sub_w)
    v_bands = [(s, e) for s, e in v_bands if em_cols < (s + e) // 2 < sub_w - em_cols]

    if v_bands:
        v_bands_abs = [(x0 + s, x0 + e) for s, e in v_bands]
        # Each band contributes a left-edge path (section boundary on the left side)
        # and a right-edge path (boundary on the right side). This excludes the band
        # pixels from both neighboring sections, preventing recursive re-detection.
        band_left_edges = []
        band_right_edges = []
        for bs, be in v_bands_abs:
            le, re = _trace_vertical_band_edges(dark, y0, y1, bs, be + 1)
            band_left_edges.append(le)
            band_right_edges.append(re)

        left_bounds = [np.full(sub_h, x0, dtype=np.int32)] + band_right_edges
        right_bounds = band_left_edges + [np.full(sub_h, x1, dtype=np.int32)]

        sections: list[Polygon] = []
        for left_path, right_path in zip(left_bounds, right_bounds):
            strip = _vertical_strip_polygon(left_path, right_path, y0, y1)
            strip_region = _safe_intersection(region, strip)
            if strip_region is None:
                continue
            sections.extend(
                _partition_polygon(
                    dark,
                    strip_region,
                    min_run_fraction,
                    min_thickness,
                    edge_margin_fraction,
                    depth + 1,
                )
            )
        return sections

    # --- Horizontal split lines ---
    min_hrun = int(min_run_fraction * sub_w)
    row_profile = _row_max_runs(sub)
    h_bands = find_thick_bands(row_profile, min_hrun, min_thickness)
    em_rows = int(edge_margin_fraction * sub_h)
    h_bands = [(s, e) for s, e in h_bands if em_rows < (s + e) // 2 < sub_h - em_rows]

    if h_bands:
        h_bands_abs = [(y0 + s, y0 + e) for s, e in h_bands]
        band_top_edges = []
        band_bot_edges = []
        for hs, he in h_bands_abs:
            te, be = _trace_horizontal_band_edges(dark, hs, he + 1, x0, x1)
            band_top_edges.append(te)
            band_bot_edges.append(be)

        top_bounds = [np.full(sub_w, y0, dtype=np.int32)] + band_bot_edges
        bot_bounds = band_top_edges + [np.full(sub_w, y1, dtype=np.int32)]

        sections = []
        for top_path, bot_path in zip(top_bounds, bot_bounds):
            strip = _horizontal_strip_polygon(top_path, bot_path, x0, x1)
            strip_region = _safe_intersection(region, strip)
            if strip_region is None:
                continue
            sections.extend(
                _partition_polygon(
                    dark,
                    strip_region,
                    min_run_fraction,
                    min_thickness,
                    edge_margin_fraction,
                    depth + 1,
                )
            )
        return sections

    return [region]


def split_image(image_path: Path) -> list[Path]:
    """Detect split lines in an image and write each section as a __N.raw.jpg file.

    Loads the image at 1/DOWNSAMPLE resolution for analysis, then crops the
    full-resolution image for each detected section. Output files are written
    next to the input file with the naming convention <stem>__1.raw.jpg,
    <stem>__2.raw.jpg, etc.

    Returns:
        List of output paths written. Empty when no splits are detected.
    """
    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
        ds_img = img.convert("L").reduce(DOWNSAMPLE)

    ds_arr = np.array(ds_img, dtype=np.uint8)
    ds_height, ds_width = ds_arr.shape

    sections_ds = partition_image(ds_arr)
    if len(sections_ds) <= 1:
        return []

    # Scale polygons from downsampled to full-resolution coordinates.
    scale_x = orig_width / ds_width
    scale_y = orig_height / ds_height
    sections_full = [
        shapely_scale(poly, xfact=scale_x, yfact=scale_y, origin=(0, 0))
        for poly in sections_ds
    ]

    stem = image_stem(str(image_path))
    output_dir = image_path.parent
    output_paths: list[Path] = []

    with Image.open(image_path) as img:
        for n, poly in enumerate(sections_full, start=1):
            bx0, by0, bx1, by1 = (
                max(0, int(poly.bounds[0])),
                max(0, int(poly.bounds[1])),
                min(orig_width, int(poly.bounds[2] + 0.5)),
                min(orig_height, int(poly.bounds[3] + 0.5)),
            )
            section = img.crop((bx0, by0, bx1, by1))
            # White-fill any pixels outside the (potentially non-rectangular) polygon.
            mask = Image.new("L", section.size, 0)
            coords = [(x - bx0, y - by0) for x, y in poly.exterior.coords]
            ImageDraw.Draw(mask).polygon(coords, fill=255)
            white = Image.new("RGB", section.size, (255, 255, 255))
            section = Image.composite(section, white, mask)
            out_path = output_dir / f"{stem}__{n}.raw.jpg"
            section.save(out_path, "JPEG", quality=95)
            output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect thick black dividing lines in Sanborn map images and write "
            "each map section as a separate JPEG file."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    args = parser.parse_args()

    for image_path_str in args.images:
        image_path = Path(image_path_str)
        output_paths = split_image(image_path)
        if output_paths:
            print(
                f"{image_path.name} → {len(output_paths)} sections",
                file=sys.stderr,
            )
            for p in output_paths:
                print(f"  {p.name}", file=sys.stderr)
        else:
            print(f"{image_path.name} → no splits detected", file=sys.stderr)


if __name__ == "__main__":
    main()
