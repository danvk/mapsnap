import type { Box, BoxesJsonData } from './types';

/**
 * Map a point from a rotated detection frame back to the original image frame.
 *
 * CRAFT runs on the image rotated by `angle` (PIL `rotate(expand=True)`), so a box's
 * coordinates are in that rotated frame. This inverts the rotation, mirroring the polygon
 * mapping in mapsnap/detect_text.py. `width`/`height` are the original image's dimensions.
 */
export function unrotatePoint(
  x: number,
  y: number,
  angle: number,
  width: number,
  height: number,
): [number, number] {
  // PIL rotate(90) is CCW; inverse: (rx, ry) -> (W-1-ry, rx).
  if (angle === 90) return [width - 1 - y, x];
  // PIL rotate(270) is CW; inverse: (rx, ry) -> (ry, H-1-rx).
  if (angle === 270) return [y, height - 1 - x];
  return [x, y];
}

// The four corners of an axis-aligned [x_min, x_max, y_min, y_max] box, clockwise.
function horizontalBoxCorners(
  box: [number, number, number, number],
): [number, number][] {
  const [xMin, xMax, yMin, yMax] = box;
  return [
    [xMin, yMin],
    [xMax, yMin],
    [xMax, yMax],
    [xMin, yMax],
  ];
}

/**
 * The longer and shorter side lengths (rounded pixels) of a 4-corner box, each taken as the
 * mean of its two opposite edges so a slightly skewed free-list quad still gets sane numbers.
 */
export function boxSideLengths(polygon: [number, number][]): {
  long: number;
  short: number;
} {
  const edge = (a: number, b: number): number =>
    Math.hypot(polygon[b][0] - polygon[a][0], polygon[b][1] - polygon[a][1]);
  const sideA = (edge(0, 1) + edge(2, 3)) / 2;
  const sideB = (edge(1, 2) + edge(3, 0)) / 2;
  return {
    long: Math.round(Math.max(sideA, sideB)),
    short: Math.round(Math.min(sideA, sideB)),
  };
}

/**
 * Flatten boxes.json's per-rotation groups into one list of boxes, each polygon mapped
 * into the original image frame and tagged with the rotation that found it.
 *
 * Both the axis-aligned (`horizontal_list`) and free-quadrilateral (`free_list`) boxes are
 * included, since together they are all of a rotation pass's detections.
 */
export function boxesFromJson(data: BoxesJsonData): Box[] {
  const boxes: Box[] = [];
  for (const group of data.boxes) {
    const angle = group.angle;
    const unrotate = (x: number, y: number): [number, number] =>
      unrotatePoint(x, y, angle, data.width, data.height);
    const add = (polygon: [number, number][]): void => {
      const { long, short } = boxSideLengths(polygon);
      boxes.push({ polygon, angle, long_side: long, short_side: short });
    };
    for (const box of group.horizontal_list) {
      add(horizontalBoxCorners(box).map(([x, y]) => unrotate(x, y)));
    }
    for (const polygon of group.free_list) {
      add(polygon.map(([x, y]) => unrotate(x, y)));
    }
  }
  return boxes;
}

/** A box paired with its index in the original, unfiltered list. */
export interface IndexedBox {
  box: Box;
  i: number;
}

/** A box's area in square pixels, used to order the detection list. */
export function boxArea(box: Box): number {
  return box.long_side * box.short_side;
}

/**
 * Return boxes passing the short/long-side minimums, paired with their original indices.
 * (Confidence, the streets view's other filter, does not exist before recognition.)
 */
export function filterBoxes(
  boxes: Box[],
  filters: { minShortSide: number; minLongSide: number },
): IndexedBox[] {
  return boxes
    .map((box, i) => ({ box, i }))
    .filter(
      ({ box }) =>
        box.short_side >= filters.minShortSide &&
        box.long_side >= filters.minLongSide,
    );
}

/** Distinct outline color for a detection rotation (0/90/270°). */
export function boxAngleColor(angle: number): string {
  switch (angle) {
    case 0:
      return '#2563eb'; // blue
    case 90:
      return '#059669'; // green
    case 270:
      return '#d97706'; // amber
    default:
      return `hsl(${(angle * 2) % 360}, 70%, 45%)`;
  }
}
