import type { AdjacencyDetection, Detection } from './types';

/** Filter sliders/toggles that control which detections are shown. */
export interface DetectionFilters {
  minConfidence: number;
  minShortSide: number;
  minLongSide: number;
  showIgnored: boolean;
}

/** A detection paired with its index in the original, unfiltered list. */
export interface IndexedDetection {
  det: Detection;
  i: number;
}

/** Map confidence in [0, 1] to a CSS color string (red → yellow → green). */
export function confidenceColor(confidence: number): string {
  const hue = Math.round(confidence * 120); // 0 = red, 120 = green
  return `hsl(${hue}, 90%, 45%)`;
}

/**
 * Hue range georeferencing refuses to read as a building fill.
 *
 * Mirrors `FILL_YELLOW_HUE_BAND` in mapsnap/georef_from_labels.py — keep the two in step.
 * Aged paper, the tape patches pasted over renamed streets, and brown ink all land in this
 * band, so a label on a yellow/brown background is ambiguous and is kept.
 */
export const FILL_YELLOW_HUE_BAND: [number, number] = [40, 110];

/**
 * True if georeferencing will discard this detection as a building label.
 *
 * A detection carries a `background` when it sits on something more saturated than the page's
 * paper, but only one outside the yellow/brown band — the red brick and blue stone of the
 * Sanborn colour code — is treated as a building.
 */
export function isOnBuildingFill(det: Detection): boolean {
  if (!det.background) return false;
  const [low, high] = FILL_YELLOW_HUE_BAND;
  return det.background.hue < low || det.background.hue > high;
}

/**
 * Convert an adjacency.json digit read to the Detection shape the overlay,
 * table, and preview canvas render. Side lengths come from the polygon's
 * bounding box; the printed sheet numbers are upright, so angle is 0.
 *
 * `mutualNumbers` holds the page numbers of this page's reciprocated
 * neighbors: a claim of one of those renders blue (`mutual: true`), any other
 * claim amber (`mutual: false`), and non-claims grey (`ignore`, no `mutual`).
 */
export function detectionFromAdjacency(
  adjacencyDetection: AdjacencyDetection,
  mutualNumbers: Set<number>,
): Detection {
  const xs = adjacencyDetection.polygon.map(([x]) => x);
  const ys = adjacencyDetection.polygon.map(([, y]) => y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  return {
    polygon: adjacencyDetection.polygon,
    text: String(adjacencyDetection.number),
    confidence: adjacencyDetection.confidence,
    angle: 0,
    long_side: Math.max(width, height),
    short_side: Math.min(width, height),
    ignore: !adjacencyDetection.claim,
    ...(adjacencyDetection.claim
      ? { mutual: mutualNumbers.has(adjacencyDetection.number) }
      : {}),
  };
}

/** Return detections passing the given filters, paired with their original indices. */
export function filterDetections(
  detections: Detection[],
  filters: DetectionFilters,
): IndexedDetection[] {
  return detections
    .map((det, i) => ({ det, i }))
    .filter(
      ({ det }) =>
        det.confidence >= filters.minConfidence &&
        det.short_side >= filters.minShortSide &&
        det.long_side >= filters.minLongSide &&
        (!det.ignore || filters.showIgnored),
    );
}

/**
 * Display rotation for a detection's preview thumbnail.
 *
 * Honors the detection's reading direction (`angle`, degrees CW), then snaps by the
 * smallest rotation that leaves the box axis-aligned — which may put the long OR the short
 * side along the horizontal, whichever is the smaller turn. So a number whose box is taller
 * than wide stays upright instead of being rotated a quarter turn, while a diagonal street
 * is straightened the short way.
 *
 * Returns the rotation (radians, image space; apply as `ctx.rotate(-textAngle)`) and whether
 * the long side ends up horizontal (used to size the canvas).
 */
export function previewOrientation(
  polygon: [number, number][],
  angle: number,
): { textAngle: number; longHorizontal: boolean } {
  // Direction of the longest polygon side = the box's orientation in the image.
  let maxLen = 0;
  let longDx = 1;
  let longDy = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % 4];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len > maxLen) {
      maxLen = len;
      longDx = dx;
      longDy = dy;
    }
  }
  const rawAngle = Math.atan2(longDy, longDx);
  // `angle` is CW, which matches image space (y-down), so the desired text direction is +angle.
  const target = (angle * Math.PI) / 180;
  const quarter = Math.PI / 2;
  // Snap the box orientation to the axis-aligned grid point nearest the target direction.
  const k = Math.round((target - rawAngle) / quarter);
  const snapped = rawAngle + k * quarter;
  return {
    // Normalize to (-π, π] so equivalent rotations compare equal.
    textAngle: Math.atan2(Math.sin(snapped), Math.cos(snapped)),
    // After rotating, the long side lies at -k*90°: horizontal for even k, vertical for odd.
    longHorizontal: ((k % 2) + 2) % 2 === 0,
  };
}

/**
 * The oriented-box geometry {@link drawDetectionCanvas} needs to crop and deskew a patch.
 * Satisfied by both a {@link Detection} and a boxes.json `Box`.
 */
export interface OrientedBox {
  polygon: [number, number][];
  angle: number;
  long_side: number;
  short_side: number;
}

/** Options for {@link drawDetectionCanvas}. */
export interface DrawDetectionOptions {
  canvas: HTMLCanvasElement;
  det: OrientedBox;
  image: HTMLImageElement;
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * Draw the rotated image patch for a detection into a canvas.
 *
 * Orientation comes from {@link previewOrientation}: the box is straightened by the
 * smallest rotation, so wide text reads horizontally while a tall number stays upright.
 */
export function drawDetectionCanvas(options: DrawDetectionOptions): void {
  const { canvas, det, image, jsonWidth, jsonHeight } = options;
  if (!image.naturalWidth) return;
  const cx = det.polygon.reduce((s, [x]) => s + x, 0) / 4;
  const cy = det.polygon.reduce((s, [, y]) => s + y, 0) / 4;

  const { textAngle, longHorizontal } = previewOrientation(
    det.polygon,
    det.angle,
  );

  // The side that ends up horizontal sets the canvas width; the other its height. Keeping
  // the vertical side at 40px (capped at 200px wide) makes a horizontal thumbnail for text
  // and an upright one for numbers.
  const horizontalSide = longHorizontal ? det.long_side : det.short_side;
  const verticalSide = longHorizontal ? det.short_side : det.long_side;

  let scale = 40 / verticalSide;
  let cW = Math.round(horizontalSide * scale);
  let cH = 40;
  if (cW > 200) {
    scale = 200 / horizontalSide;
    cW = 200;
    cH = Math.round(verticalSide * scale);
  }
  canvas.width = cW;
  canvas.height = cH;

  // When the loaded image is larger than the json coordinate space (e.g. full-res
  // image with streets.json from a downscaled version), render the image at
  // jsonWidth*scale × jsonHeight*scale so one json-pixel equals one canvas pixel at
  // the given scale, keeping the crop center at the polygon centroid.
  const ctx = canvas.getContext('2d')!;
  ctx.save();
  ctx.translate(cW / 2, cH / 2);
  ctx.rotate(-textAngle);
  ctx.drawImage(
    image,
    -cx * scale,
    -cy * scale,
    jsonWidth * scale,
    jsonHeight * scale,
  );
  ctx.restore();
}
