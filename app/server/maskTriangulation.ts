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

const roundTenth = (value: number) => Math.round(value * 10) / 10;

/**
 * The maps in an annotation that Allmaps fails to triangulate, as the viewer draws them.
 *
 * Allmaps triangulates each map's mask with its GCPs as fixed points, and throws
 * ("Constraining edge intersects point", "Edge intersects already constrained edge")
 * when a GCP sits on a mask edge or mask edges overlap. The viewer and the published
 * CDN copies draw quarter-size images, so the mask and GCPs are first scaled by
 * `scale`, clamped to the image and rounded to 0.1 px, as they are there. Our maps
 * use linear transformations, for which Allmaps triangulates at its default
 * resolution; that is the one tried here.
 */
export function triangulationFailures(
  annotation: unknown,
  scale = 0.25,
): TriangulationFailure[] {
  const failures: TriangulationFailure[] = [];
  parseAnnotation(annotation).forEach((map, index) => {
    const { width: fullWidth, height: fullHeight } = map.resource;
    if (!fullWidth || !fullHeight) {
      failures.push({ index, message: 'image has no dimensions' });
      return;
    }
    const width = Math.ceil(fullWidth * scale);
    const height = Math.ceil(fullHeight * scale);
    const sx = width / fullWidth;
    const sy = height / fullHeight;
    const mask: Point[] = map.resourceMask.map(([x, y]) => [
      roundTenth(Math.min(Math.max(x * sx, 0), width)),
      roundTenth(Math.min(Math.max(y * sy, 0), height)),
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
