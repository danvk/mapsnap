import { parseAnnotation } from '@allmaps/annotation';
import { triangulateToUnique } from '@allmaps/triangulate';
import {
  bboxToPolygon,
  bufferBbox,
  combineBboxes,
  computeBbox,
} from '@allmaps/stdlib';

type Point = [number, number];

/** One map that Allmaps could not triangulate. */
export interface TriangulationFailure {
  /** Position of the map in the annotation's items. */
  index: number;
  message: string;
}

/** How a viewer receives a map: at what image scale, and with what rounding noise. */
export interface Rendering {
  /** Image size relative to the annotation's canvas (0.25 for our quarter-size copies). */
  scale: number;
  /** Largest random offset (px) added to each mask vertex, standing in for rounding. */
  jitter: number;
  /** Seed for the jitter. */
  seed: number;
}

/**
 * The renderings a map meets in practice.
 *
 * loc.gov's tile service serves the full canvas; the volume viewer and the
 * published CDN copies serve quarter-size images. Each is tried as written and
 * with two small jitters: Allmaps' failures are numeric coincidences, so a mask
 * that only just triangulates (a sliver, a vertex grazing a GCP) fails some of them.
 */
export const RENDERINGS: Rendering[] = [1, 0.25].flatMap((scale) => [
  { scale, jitter: 0, seed: 0 },
  { scale, jitter: 0.05, seed: 1 },
  { scale, jitter: 0.05, seed: 2 },
]);

const roundTenth = (value: number) => Math.round(value * 10) / 10;

/** A deterministic pseudo-random sequence in [-1, 1). */
function noise(seed: number): () => number {
  let state = (seed * 2654435761) % 4294967296 || 1;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state / 4294967296) * 2 - 1;
  };
}

/**
 * The maps in an annotation that Allmaps fails to triangulate under one rendering.
 *
 * Allmaps triangulates each map's mask with its GCPs as fixed points, and throws
 * ("Constraining edge intersects point", "Edge intersects already constrained edge")
 * when a GCP sits on a mask edge or mask edges overlap. The mask and GCPs are
 * scaled, clamped to the image and rounded to 0.1 px, as our image rewrites do,
 * then the mask's vertices are jittered. Our maps use linear transformations, for
 * which Allmaps triangulates at its default resolution; that is the one tried here.
 */
export function triangulationFailures(
  annotation: unknown,
  rendering: Rendering = { scale: 0.25, jitter: 0, seed: 0 },
): TriangulationFailure[] {
  const failures: TriangulationFailure[] = [];
  const random = noise(rendering.seed);
  parseAnnotation(annotation).forEach((map, index) => {
    const { width: fullWidth, height: fullHeight } = map.resource;
    if (!fullWidth || !fullHeight) {
      failures.push({ index, message: 'image has no dimensions' });
      return;
    }
    const width = Math.ceil(fullWidth * rendering.scale);
    const height = Math.ceil(fullHeight * rendering.scale);
    const sx = width / fullWidth;
    const sy = height / fullHeight;
    const mask: Point[] = map.resourceMask.map(([x, y]) => [
      roundTenth(Math.min(Math.max(x * sx, 0), width)) +
        rendering.jitter * random(),
      roundTenth(Math.min(Math.max(y * sy, 0), height)) +
        rendering.jitter * random(),
    ]);
    const gcps: Point[] = map.gcps.map((gcp) => [
      roundTenth(gcp.resource[0] * sx),
      roundTenth(gcp.resource[1] * sy),
    ]);
    const image: Point[] = [
      [0, 0],
      [width, 0],
      [width, height],
      [0, height],
    ];
    const bbox = combineBboxes(
      computeBbox(image),
      bufferBbox(computeBbox(mask), 1),
    );
    if (!bbox) {
      failures.push({ index, message: 'mask has no extent' });
      return;
    }
    try {
      triangulateToUnique(bboxToPolygon(bbox), undefined, {
        steinerPoints: gcps,
        steinerPolygons: [[mask]],
        computeInsideSteinerPolygons: true,
      });
    } catch (error) {
      failures.push({ index, message: String((error as Error).message) });
    }
  });
  return failures;
}

/**
 * Each map's share of `renderings` that Allmaps fails to triangulate, by item position.
 *
 * A map at 0 draws everywhere; one at 1 never draws; one in between is fragile.
 */
export function failureRates(
  annotation: unknown,
  renderings: Rendering[] = RENDERINGS,
): number[] {
  const maps = parseAnnotation(annotation).length;
  const counts = new Array<number>(maps).fill(0);
  for (const rendering of renderings) {
    for (const { index } of triangulationFailures(annotation, rendering)) {
      counts[index] += 1;
    }
  }
  return counts.map((count) => count / renderings.length);
}
