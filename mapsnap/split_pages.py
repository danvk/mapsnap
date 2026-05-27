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
from PIL import Image

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


def partition_image(
    img_ds: np.ndarray,
    min_run_fraction: float = MIN_RUN_FRACTION,
    min_thickness: int = MIN_THICKNESS_PX,
    edge_margin_fraction: float = EDGE_MARGIN_FRACTION,
) -> list[tuple[int, int, int, int]]:
    """Recursively partition a downsampled grayscale image into map sections.

    Detects thick black dividing lines by searching for rows or columns containing
    long consecutive dark-pixel runs, then crops away each detected band and recurses
    on the remaining sections. Vertical splits are processed first (left-to-right),
    then horizontal splits within each resulting column (top-to-bottom), handling
    T-shaped layouts naturally.

    Args:
        img_ds: Downsampled grayscale image as a 2-D uint8 array (H, W).
        min_run_fraction: A row/column qualifies when its longest dark run exceeds
                          this fraction of the relevant dimension.
        min_thickness: Minimum band width in pixels (downsampled coordinates).
        edge_margin_fraction: Bands whose center is within this fraction of the image
                              edge are ignored (they are page border frames).

    Returns:
        List of (x0, y0, x1, y1) rectangles in downsampled image coordinates,
        ordered left-to-right then top-to-bottom.
    """
    dark = img_ds < DARK_THRESHOLD
    h, w = dark.shape
    return _partition(
        dark, 0, 0, w, h, min_run_fraction, min_thickness, edge_margin_fraction, depth=0
    )


def _partition(
    dark: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_run_fraction: float,
    min_thickness: int,
    edge_margin_fraction: float,
    depth: int,
) -> list[tuple[int, int, int, int]]:
    """Recursively split the sub-region dark[y0:y1, x0:x1] along thick dark bands."""
    if depth >= 3 or x1 <= x0 or y1 <= y0:
        return [(x0, y0, x1, y1)]

    sub = dark[y0:y1, x0:x1]
    sub_h, sub_w = sub.shape

    # Check for vertical split lines: columns with long vertical dark runs.
    min_vrun = int(min_run_fraction * sub_h)
    col_profile = _col_max_runs(sub)
    v_bands = find_thick_bands(col_profile, min_vrun, min_thickness)
    # Discard bands too close to the image edge (page border frames).
    em_cols = int(edge_margin_fraction * sub_w)
    v_bands = [(s, e) for s, e in v_bands if em_cols < (s + e) // 2 < sub_w - em_cols]

    if v_bands:
        sections: list[tuple[int, int, int, int]] = []
        cur_x = x0
        for b_start, b_end in v_bands:
            abs_b_end = x0 + b_end
            if cur_x < x0 + b_start:
                sections.extend(
                    _partition(
                        dark,
                        cur_x,
                        y0,
                        x0 + b_start,
                        y1,
                        min_run_fraction,
                        min_thickness,
                        edge_margin_fraction,
                        depth + 1,
                    )
                )
            cur_x = abs_b_end + 1
        if cur_x < x1:
            sections.extend(
                _partition(
                    dark,
                    cur_x,
                    y0,
                    x1,
                    y1,
                    min_run_fraction,
                    min_thickness,
                    edge_margin_fraction,
                    depth + 1,
                )
            )
        return sections

    # Check for horizontal split lines: rows with long horizontal dark runs.
    min_hrun = int(min_run_fraction * sub_w)
    row_profile = _row_max_runs(sub)
    h_bands = find_thick_bands(row_profile, min_hrun, min_thickness)
    em_rows = int(edge_margin_fraction * sub_h)
    h_bands = [(s, e) for s, e in h_bands if em_rows < (s + e) // 2 < sub_h - em_rows]

    if h_bands:
        sections = []
        cur_y = y0
        for b_start, b_end in h_bands:
            abs_b_end = y0 + b_end
            if cur_y < y0 + b_start:
                sections.extend(
                    _partition(
                        dark,
                        x0,
                        cur_y,
                        x1,
                        y0 + b_start,
                        min_run_fraction,
                        min_thickness,
                        edge_margin_fraction,
                        depth + 1,
                    )
                )
            cur_y = abs_b_end + 1
        if cur_y < y1:
            sections.extend(
                _partition(
                    dark,
                    x0,
                    cur_y,
                    x1,
                    y1,
                    min_run_fraction,
                    min_thickness,
                    edge_margin_fraction,
                    depth + 1,
                )
            )
        return sections

    return [(x0, y0, x1, y1)]


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

    # Map downsampled coordinates back to full-resolution.
    scale_x = orig_width / ds_width
    scale_y = orig_height / ds_height
    sections_full = []
    for x0_ds, y0_ds, x1_ds, y1_ds in sections_ds:
        x0 = max(0, min(round(x0_ds * scale_x), orig_width))
        y0 = max(0, min(round(y0_ds * scale_y), orig_height))
        x1 = max(0, min(round(x1_ds * scale_x), orig_width))
        y1 = max(0, min(round(y1_ds * scale_y), orig_height))
        sections_full.append((x0, y0, x1, y1))

    stem = image_stem(str(image_path))
    output_dir = image_path.parent
    output_paths: list[Path] = []

    with Image.open(image_path) as img:
        for n, (x0, y0, x1, y1) in enumerate(sections_full, start=1):
            section = img.crop((x0, y0, x1, y1))
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
