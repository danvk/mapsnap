import type { Detection } from './types';

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

/** Options for {@link drawDetectionCanvas}. */
export interface DrawDetectionOptions {
  canvas: HTMLCanvasElement;
  det: Detection;
  image: HTMLImageElement;
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * Draw the rotated image patch for a detection into a canvas.
 *
 * Rotation is derived from the direction of the polygon's longest side so that
 * diagonally-oriented text is also rendered horizontally.
 */
export function drawDetectionCanvas(options: DrawDetectionOptions): void {
  const { canvas, det, image, jsonWidth, jsonHeight } = options;
  if (!image.naturalWidth) return;
  const cx = det.polygon.reduce((s, [x]) => s + x, 0) / 4;
  const cy = det.polygon.reduce((s, [, y]) => s + y, 0) / 4;

  // Find the direction of the longest polygon side to determine text orientation.
  let maxLen = 0;
  let longDx = 1,
    longDy = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = det.polygon[i];
    const [x2, y2] = det.polygon[(i + 1) % 4];
    const dx = x2 - x1,
      dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > maxLen) {
      maxLen = len;
      longDx = dx;
      longDy = dy;
    }
  }
  // Use det.angle (0/90/270) to disambiguate the 180° ambiguity of the long-side direction.
  // 90° and 270° both indicate vertical text; folding to the smaller angle gives the same
  // reference direction (-π/2) for both, so the disambiguation picks the correct half.
  const rawAngle = Math.atan2(longDy, longDx);
  const foldedAngle = Math.min(det.angle, 360 - det.angle);
  const octantAngle = (-foldedAngle * Math.PI) / 180;
  let textAngle = rawAngle;
  if (Math.cos(rawAngle - octantAngle) < 0) {
    // The long side points the wrong way; flip 180°.
    textAngle = rawAngle + Math.PI;
  }
  if (det.angle == 90) {
    textAngle += Math.PI;
  }

  let scale = 40 / det.short_side;
  let cW = Math.round(det.long_side * scale);
  let cH = 40;
  if (cW > 200) {
    scale = 200 / det.long_side;
    cW = 200;
    cH = Math.round(det.short_side * scale);
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
