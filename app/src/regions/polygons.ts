import { pointInPolygon, polygonArea } from '../geometry';
import type {
  KeymapDetection,
  RegionPolygon,
  RegionsWriteRequest,
} from './types';

/** Vertices a ring needs before it can be closed into a region. */
export const MIN_RING_VERTICES = 3;

/**
 * Screen-pixel radius within which a click counts as hitting a vertex, or as
 * closing the ring by returning to its first vertex. Screen rather than image
 * pixels so the target stays the same size at every zoom level.
 */
export const VERTEX_HIT_RADIUS = 9;

/**
 * Screen pixels a rectangle drag must span before it counts as a rectangle.
 *
 * Below this it was a click, not a drag — usually a mis-aimed attempt to select an
 * existing region — and turning it into a sliver polygon would be worse than nothing.
 */
export const MIN_RECT_DRAG = 6;

/**
 * Axis-aligned ring spanning two opposite corners, clockwise from the top left.
 *
 * Key-map blocks are overwhelmingly rectangles aligned with the sheet, so dragging
 * out two corners beats clicking four. The result is an ordinary four-vertex ring —
 * nothing downstream knows it was drawn differently, so editing, auto-naming and the
 * sidecar format are all unchanged.
 */
export function rectRing(
  a: [number, number],
  b: [number, number],
): [number, number][] {
  const x0 = Math.min(a[0], b[0]);
  const x1 = Math.max(a[0], b[0]);
  const y0 = Math.min(a[1], b[1]);
  const y1 = Math.max(a[1], b[1]);
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}

/** Ring with its first vertex repeated at the end, as the panels schema wants. */
export function closedRing(ring: [number, number][]): number[][] {
  if (ring.length === 0) return [];
  const first = ring[0]!;
  const last = ring[ring.length - 1]!;
  const closed: number[][] = ring.map(([x, y]) => [round1(x), round1(y)]);
  if (first[0] !== last[0] || first[1] !== last[1]) {
    closed.push([round1(first[0]), round1(first[1])]);
  }
  return closed;
}

/** Drop a ring's duplicated closing vertex, for editing an on-disk sidecar. */
export function openRing(ring: number[][]): [number, number][] {
  const points: [number, number][] = ring.map(([x = 0, y = 0]) => [x, y]);
  if (points.length < 2) return points;
  const first = points[0]!;
  const last = points[points.length - 1]!;
  if (first[0] === last[0] && first[1] === last[1]) points.pop();
  return points;
}

// One decimal is the precision page_regions writes; more is false detail at this scale.
function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

/** Build the sidecar payload for a sheet. The server fills in `image`. */
export function createRegionsJson(
  width: number,
  height: number,
  regions: RegionPolygon[],
): RegionsWriteRequest {
  return {
    width,
    height,
    panels: regions.map((region) => closedRing(region.ring)),
    labels: regions.map((region) => region.text),
  };
}

/**
 * The page key a freshly drawn ring should take, from the sheet's detected numbers.
 *
 * A key map's blocks each contain exactly one printed page number, so a ring that
 * encloses exactly one detection names itself and the user types nothing. Enclosing
 * none (or several, where the detector doubled up or the ring spans two blocks) is
 * ambiguous, so it stays blank for the user to fill in rather than guessing.
 *
 * `taken` are keys already used on this sheet; a detection whose key is taken is
 * ignored, so re-drawing near an existing region does not silently duplicate its key.
 */
export function autoLabel(
  ring: [number, number][],
  detections: readonly KeymapDetection[],
  taken: ReadonlySet<string> = new Set(),
): string {
  if (ring.length < MIN_RING_VERTICES) return '';
  const enclosed = detections.filter(
    (d) =>
      d.text !== '' && !taken.has(d.text) && pointInPolygon(d.x, d.y, ring),
  );
  const keys = new Set(enclosed.map((d) => d.text));
  return keys.size === 1 ? enclosed[0]!.text : '';
}

/** Index of the region under a point, topmost (last drawn) first; null if none. */
export function regionAt(
  regions: readonly RegionPolygon[],
  x: number,
  y: number,
): number | null {
  for (let i = regions.length - 1; i >= 0; i--) {
    if (pointInPolygon(x, y, regions[i]!.ring)) return i;
  }
  return null;
}

/** A region's vertex within `radius` image px of a point, or null. */
export function vertexAt(
  region: RegionPolygon,
  x: number,
  y: number,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestDistance = radius;
  region.ring.forEach(([vx, vy], index) => {
    const distance = Math.hypot(vx - x, vy - y);
    if (distance <= bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}

/**
 * Share of the sheet a ring covers — the quantity that made a leaked segmentation
 * catastrophic, so it is worth showing while drawing.
 */
export function sheetFraction(
  ring: [number, number][],
  width: number,
  height: number,
): number {
  if (ring.length < MIN_RING_VERTICES || width <= 0 || height <= 0) return 0;
  return polygonArea(ring) / (width * height);
}
