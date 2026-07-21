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
    for (const box of group.horizontal_list) {
      boxes.push({
        polygon: horizontalBoxCorners(box).map(([x, y]) => unrotate(x, y)),
        angle,
      });
    }
    for (const polygon of group.free_list) {
      boxes.push({ polygon: polygon.map(([x, y]) => unrotate(x, y)), angle });
    }
  }
  return boxes;
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
