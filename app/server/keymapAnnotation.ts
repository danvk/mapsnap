/**
 * A georeference annotation for a key map, warped the way keymap-snap warps it.
 *
 * The key map's shipped georef is a global affine, and its residuals against
 * the sheet's own inlier intersections are large: 127 ft median on New Orleans
 * 1896, 90 ft on Detroit. `mapsnap.keymap.snap.keymap_model` therefore models
 * the sheet as a thin-plate spline through those same intersections, and that
 * model -- not the affine -- is the frame every keymap-snap placement is
 * measured in (#211). Drawing the sheet at its four affine corners would show
 * the viewer something the pipeline does not use.
 *
 * So the underlay is served as a georeference annotation carrying the sheet's
 * own GCPs with `transformation: thinPlateSpline`, which allmaps renders
 * directly. allmaps interpolates exactly, while `keymap_model` smooths
 * (TPS_SMOOTHING), and that smoothing is what keeps straight streets straight
 * between GCPs (see thinPlateSpline.ts). So the annotation's targets are not
 * the raw OSM points but the smoothed spline's prediction at each GCP pixel:
 * the exact spline allmaps fits through those reproduces the pipeline's
 * surface (measured 0.0 ft apart on New Orleans 1896).
 */
import type { GeorefAnnotationPage } from './iiifAnnotations.ts';
import {
  thinPlateSpline,
  TPS_SMOOTHING,
  type Point2,
} from './thinPlateSpline.ts';

/** One inlier intersection of a key-map georef: sheet pixel and its world point. */
export interface KeymapGcp {
  x: number;
  y: number;
  lon: number;
  lat: number;
  inlier?: boolean;
}

const M_PER_DEG_LAT = 110_540.0;
const M_PER_DEG_LON_EQUATOR = 111_320.0;
const FEET_PER_METRE = 3.28084;

/**
 * Two readings of one pixel further apart than this are a contradiction.
 * Mirrors `mapsnap.keymap.snap.GCP_CONFLICT_FT`.
 */
export const GCP_CONFLICT_FT = 50.0;

/**
 * Fewest consistent GCPs that can constrain a warp; below it the sheet is
 * drawn by its affine corners. Mirrors `road_prob.MIN_KEYMAP_INLIERS`.
 */
export const MIN_KEYMAP_GCPS = 25;

/**
 * Drop GCPs that put one key-map pixel in two different places.
 *
 * A port of `mapsnap.keymap.snap.consistent_gcps`: a pixel matched to two OSM
 * intersections more than GCP_CONFLICT_FT apart is a contradiction the spline
 * cannot resolve, and there is no evidence here for which reading is right, so
 * the whole group goes.
 */
export function consistentGcps(inliers: KeymapGcp[]): KeymapGcp[] {
  const groups = new Map<string, KeymapGcp[]>();
  for (const gcp of inliers) {
    const key = `${gcp.x.toFixed(1)},${gcp.y.toFixed(1)}`;
    const group = groups.get(key);
    if (group) group.push(gcp);
    else groups.set(key, [gcp]);
  }
  const kept: KeymapGcp[] = [];
  for (const readings of groups.values()) {
    if (readings.length === 1) {
      kept.push(readings[0]);
      continue;
    }
    const lons = readings.map((r) => r.lon);
    const lats = readings.map((r) => r.lat);
    const lonScale =
      M_PER_DEG_LON_EQUATOR * Math.cos((lats[0] * Math.PI) / 180);
    const spread =
      Math.hypot(
        (Math.max(...lons) - Math.min(...lons)) * lonScale,
        (Math.max(...lats) - Math.min(...lats)) * M_PER_DEG_LAT,
      ) * FEET_PER_METRE;
    if (spread <= GCP_CONFLICT_FT) kept.push(readings[0]);
  }
  return kept;
}

/** A georef document's four (lon, lat) corners, when it has them. */
function corners(georef: unknown): [number, number][] | undefined {
  const value = (georef as { corners?: unknown })?.corners;
  if (!Array.isArray(value) || value.length !== 4) return undefined;
  const points = value.map((corner) =>
    Array.isArray(corner) &&
    typeof corner[0] === 'number' &&
    typeof corner[1] === 'number'
      ? ([corner[0], corner[1]] as [number, number])
      : null,
  );
  return points.every((point) => point !== null)
    ? (points as [number, number][])
    : undefined;
}

/**
 * The smoothed key-map model's world point at each GCP pixel, in the frame
 * `keymap_model` uses: metres from the first GCP, with the longitude scale at
 * the GCPs' mean latitude.
 */
export function smoothedTargets(
  gcps: KeymapGcp[],
  smoothing: number = TPS_SMOOTHING,
): [number, number][] {
  const origin = [gcps[0].lon, gcps[0].lat];
  const meanLat = gcps.reduce((sum, gcp) => sum + gcp.lat, 0) / gcps.length;
  const lonScale = M_PER_DEG_LON_EQUATOR * Math.cos((meanLat * Math.PI) / 180);
  const pixels: Point2[] = gcps.map((gcp) => [gcp.x, gcp.y]);
  const metres: Point2[] = gcps.map((gcp) => [
    (gcp.lon - origin[0]) * lonScale,
    (gcp.lat - origin[1]) * M_PER_DEG_LAT,
  ]);
  return thinPlateSpline(
    pixels,
    metres,
    smoothing,
  )(pixels).map(([x, y]) => [
    x / lonScale + origin[0],
    y / M_PER_DEG_LAT + origin[1],
  ]);
}

function feature(pixel: [number, number], world: [number, number]) {
  return {
    type: 'Feature' as const,
    properties: { resourceCoords: pixel },
    geometry: { type: 'Point' as const, coordinates: world },
  };
}

/**
 * A one-item georeference AnnotationPage placing a key-map image on the world.
 *
 * `serviceUrl` is the IIIF image service for the image to draw (the sheet or
 * its P(road) map, which share the sheet's pixel frame, so the same GCPs place
 * either). Uses a thin-plate spline through the sheet's consistent inlier
 * intersections when there are enough of them, and the affine corners
 * otherwise; the returned `transformation` says which.
 */
export function keymapAnnotation(
  georef: unknown,
  serviceUrl: string,
  annotationId: string,
): GeorefAnnotationPage | null {
  const document = georef as {
    width?: number;
    height?: number;
    intersections?: KeymapGcp[];
  };
  const width = document?.width;
  const height = document?.height;
  if (!width || !height) return null;

  const gcps = consistentGcps(
    (document.intersections ?? []).filter((point) => point.inlier),
  );
  let transformation: { type: string; options?: { order: number } };
  let features: ReturnType<typeof feature>[];
  if (gcps.length >= MIN_KEYMAP_GCPS) {
    transformation = { type: 'thinPlateSpline' };
    const targets = smoothedTargets(gcps);
    features = gcps.map((gcp, i) => feature([gcp.x, gcp.y], targets[i]));
  } else {
    const ring = corners(georef);
    if (!ring) return null;
    transformation = { type: 'polynomial', options: { order: 1 } };
    const pixels: [number, number][] = [
      [0, 0],
      [width, 0],
      [width, height],
      [0, height],
    ];
    features = pixels.map((pixel, index) => feature(pixel, ring[index]));
  }

  return {
    '@context': 'http://iiif.io/api/extension/georeference/1/context.json',
    type: 'AnnotationPage',
    items: [
      {
        id: annotationId,
        type: 'Annotation',
        motivation: 'georeferencing',
        target: {
          type: 'SpecificResource',
          source: {
            id: serviceUrl,
            type: 'ImageService3',
            width,
            height,
          },
          // Required by the georeference-annotation schema, and the mask
          // allmaps clips the image to: the whole sheet, margins included.
          selector: {
            type: 'SvgSelector',
            value: `<svg width="${width}" height="${height}"><polygon points="0,0 ${width},0 ${width},${height} 0,${height}" /></svg>`,
          },
        },
        body: {
          type: 'FeatureCollection',
          transformation,
          features,
        },
      },
    ],
  } as unknown as GeorefAnnotationPage;
}
