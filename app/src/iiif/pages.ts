/**
 * Per-page geometry derived from a rewritten Georeference AnnotationPage.
 *
 * The volume viewer needs, for each page: its geographic footprint (full image
 * rectangle and clipping polygon) for hit-testing and selection outlines, plus
 * scale and rotation stats for the info panel. Everything is computed from the
 * annotation's GCPs with the same affine model the pipeline fits, so it lines
 * up with what the Allmaps layer renders.
 */

import type {
  GeorefAnnotationPage,
  GcpFeature,
} from '../../server/iiifAnnotations';
import { computeCorners, projectThroughCorners } from '../geometry';
import type { Corners, Street } from '../types';

const FEET_PER_METER = 3.28084;
const METERS_PER_DEGREE_LAT = 110574;
const METERS_PER_DEGREE_LON_AT_EQUATOR = 111320;

/** One of a page's GCPs: image pixel position, geo position, and kind. */
export interface PageGcp {
  x: number;
  y: number;
  lon: number;
  lat: number;
  /** The annotation's GCP kind: "gcp" (real) or "corner" (fallback). */
  type: string;
}

/** A page's derived geometry and stats, ready for map display and the info panel. */
export interface PageGeo {
  /** Index of this page's item in the annotation's items array; selection id. */
  itemIndex: number;
  /** Parent page key (splits share it), e.g. "p1499m" for both p1499m panels. */
  pageKey: string;
  /**
   * Split panel number from the annotation id (`…__2/georef`), or null for a whole
   * page. Present only on our generated splits; a truth split carries it in its
   * label instead and is matched by panel overlap rather than by number.
   */
  splitIndex: number | null;
  /** Page key including any split index, matching the on-disk file stem (e.g. "p1499m__2"). */
  stem: string;
  width: number;
  height: number;
  /** Geo images of the image corners (0,0), (w,0), (w,h), (0,h) as [lon, lat]. */
  corners: Corners;
  /** Closed [lon, lat] ring of the full image rectangle. */
  rectRing: [number, number][];
  /** Closed [lon, lat] ring of the clipping polygon. */
  clipRing: [number, number][];
  scalePixelsPerFoot: number;
  /** Rotation from north-up in degrees, positive clockwise. */
  rotationDegrees: number;
  gcps: PageGcp[];
  /** The annotation's transformation type, e.g. "polynomial" or "helmert". */
  transformationType: string;
}

/**
 * Split panel number for an item, or null for a whole page.
 *
 * A generated split carries it in its id (`…-1499M__2/georef`); a truth split carries it in
 * a trailing `[N]` on its label. A generated whole page's id ends in `/georef` and its label
 * may still carry a stray `[N]` copied from the truth — which is ignored.
 */
function splitIndexFor(
  id: string | undefined,
  label: string | undefined,
): number | null {
  const idMatch = id?.match(/__(\d+)\//);
  if (idMatch) return Number(idMatch[1]);
  if (id?.includes('/georef')) return null; // generated whole page
  const labelMatch = label?.match(/\[(\d+)\]\s*$/);
  return labelMatch ? Number(labelMatch[1]) : null;
}

/** Parse the vertex list out of an SvgSelector's polygon value. */
export function svgPolygonPoints(svg: string): [number, number][] {
  const match = svg.match(/points="([^"]*)"/);
  if (!match || !match[1]) return [];
  return match[1]
    .trim()
    .split(/\s+/)
    .map((pair) => {
      const [x = 0, y = 0] = pair.split(',').map(Number);
      return [x, y];
    });
}

// East/north displacement in meters from a to b, in a local equirectangular
// frame — fine at page scale, and consistent with the helmert fit below.
function deltaMeters(
  a: [number, number],
  b: [number, number],
): [number, number] {
  const latRefRadians = (((a[1] + b[1]) / 2) * Math.PI) / 180;
  return [
    (b[0] - a[0]) * Math.cos(latRefRadians) * METERS_PER_DEGREE_LON_AT_EQUATOR,
    (b[1] - a[1]) * METERS_PER_DEGREE_LAT,
  ];
}

/** Bearing of the a→b geo vector in degrees clockwise from north. */
export function bearingDegrees(
  a: [number, number],
  b: [number, number],
): number {
  const [eastMeters, northMeters] = deltaMeters(a, b);
  return (Math.atan2(eastMeters, northMeters) * 180) / Math.PI;
}

/**
 * Exact similarity (helmert) fit through two GCPs, mapping image pixels
 * (y down) to geo coordinates; returns the images of the four page corners.
 *
 * Works in a local meter frame so the fit is conformal despite the unequal
 * meters-per-degree of longitude and latitude. The reflected similarity form
 * (E = c·x + d·y, N = d·x − c·y) absorbs the image's y-down handedness.
 */
function helmertCorners(
  points: [PageGcp, PageGcp],
  width: number,
  height: number,
): Corners | null {
  const [p, q] = points;
  const lonRef = (p.lon + q.lon) / 2;
  const latRef = (p.lat + q.lat) / 2;
  const metersPerLon =
    Math.cos((latRef * Math.PI) / 180) * METERS_PER_DEGREE_LON_AT_EQUATOR;
  const toMeters = (lon: number, lat: number): [number, number] => [
    (lon - lonRef) * metersPerLon,
    (lat - latRef) * METERS_PER_DEGREE_LAT,
  ];
  const [pEast, pNorth] = toMeters(p.lon, p.lat);
  const [qEast, qNorth] = toMeters(q.lon, q.lat);
  const deltaX = q.x - p.x;
  const deltaY = q.y - p.y;
  const denominator = deltaX * deltaX + deltaY * deltaY;
  if (denominator === 0) return null;
  const deltaEast = qEast - pEast;
  const deltaNorth = qNorth - pNorth;
  const c = (deltaEast * deltaX - deltaNorth * deltaY) / denominator;
  const d = (deltaEast * deltaY + deltaNorth * deltaX) / denominator;
  const translateEast = pEast - c * p.x - d * p.y;
  const translateNorth = pNorth - d * p.x + c * p.y;
  const transform = (x: number, y: number): [number, number] => [
    lonRef + (c * x + d * y + translateEast) / metersPerLon,
    latRef + (d * x - c * y + translateNorth) / METERS_PER_DEGREE_LAT,
  ];
  return [
    transform(0, 0),
    transform(width, 0),
    transform(width, height),
    transform(0, height),
  ];
}

// Extract usable GCPs (image pixel + geo coordinates) from an item's features.
function gcpPoints(features: GcpFeature[]): PageGcp[] {
  const points: PageGcp[] = [];
  for (const feature of features) {
    const resourceCoords = feature.properties?.resourceCoords;
    const geoCoords = (
      feature.geometry as { coordinates?: number[] } | undefined
    )?.coordinates;
    if (
      resourceCoords &&
      resourceCoords.length >= 2 &&
      geoCoords &&
      geoCoords.length >= 2
    ) {
      points.push({
        x: resourceCoords[0] ?? 0,
        y: resourceCoords[1] ?? 0,
        lon: geoCoords[0] ?? 0,
        lat: geoCoords[1] ?? 0,
        type: String(feature.properties?.type ?? 'gcp'),
      });
    }
  }
  return points;
}

/**
 * Truth pages that the loaded run never georeferenced, ready to show as
 * "missing" rows and footprints.
 *
 * A truth page is missing when its page key is absent from the fitted pages.
 * Results are deduped by page key (a split parent has one truth item per panel;
 * we keep the first) and given negative `itemIndex` selection ids so they never
 * collide with the fitted pages' array indices.
 */
export function missingTruthPages(
  fitPages: PageGeo[],
  truthPages: PageGeo[],
): PageGeo[] {
  const fitKeys = new Set(fitPages.map((page) => page.pageKey));
  const seen = new Set<string>();
  const missing: PageGeo[] = [];
  for (const truthPage of truthPages) {
    if (fitKeys.has(truthPage.pageKey) || seen.has(truthPage.pageKey)) continue;
    seen.add(truthPage.pageKey);
    // The whole page is missing (no panel was fitted), so label it by the parent
    // key rather than the first truth panel's split stem.
    missing.push({
      ...truthPage,
      itemIndex: -(missing.length + 1),
      splitIndex: null,
      stem: truthPage.pageKey,
    });
  }
  return missing;
}

// Close a ring in place if its last point differs from its first.
function closedRing(ring: [number, number][]): [number, number][] {
  if (ring.length === 0) return ring;
  const first = ring[0]!;
  const last = ring[ring.length - 1]!;
  if (first[0] !== last[0] || first[1] !== last[1]) {
    return [...ring, first];
  }
  return ring;
}

/**
 * Derive geometry and stats for every page in a rewritten AnnotationPage.
 *
 * Items without a `page` metadata entry or enough GCPs to fit a transform are
 * skipped. `itemIndex` records each page's position in `annotation.items`, so
 * results can be matched to the map IDs returned by addGeoreferenceAnnotation.
 */
export function pagesFromAnnotation(
  annotation: GeorefAnnotationPage,
): PageGeo[] {
  const pages: PageGeo[] = [];
  (annotation.items ?? []).forEach((item, itemIndex) => {
    const source = item.target?.source;
    const pageKey = item.metadata?.find(
      (entry) => entry.label === 'page',
    )?.value;
    if (!source?.width || !source.height || !pageKey) return;
    const { width, height } = source;

    const points = gcpPoints(item.body?.features ?? []);
    let corners: Corners | null = null;
    if (points.length >= 3) {
      const asStreets: Street[] = points.map((p) => ({ street: '', ...p }));
      corners = computeCorners(asStreets, width, height);
    } else if (points.length === 2) {
      corners = helmertCorners([points[0]!, points[1]!], width, height);
    }
    if (!corners) return;

    const rectRing = closedRing([...corners]);
    const clipPoints = svgPolygonPoints(item.target?.selector?.value ?? '');
    const clipRing =
      clipPoints.length >= 3
        ? closedRing(
            clipPoints.map(([x, y]) =>
              projectThroughCorners(corners, width, height, x, y),
            ),
          )
        : rectRing;

    const splitIndex = splitIndexFor(item.id, item.label);
    const stem = splitIndex != null ? `${pageKey}__${splitIndex}` : pageKey;

    const [nw, ne, , sw] = corners;
    const feetAcross = Math.hypot(...deltaMeters(nw, ne)) * FEET_PER_METER;
    const feetDown = Math.hypot(...deltaMeters(nw, sw)) * FEET_PER_METER;
    if (feetAcross === 0 || feetDown === 0) return;
    const scalePixelsPerFoot = (width / feetAcross + height / feetDown) / 2;
    const bearing = bearingDegrees(nw, ne);
    const rotationDegrees = ((bearing - 90 + 540) % 360) - 180;

    const transformation = item.body?.transformation as
      | { type?: string }
      | undefined;
    pages.push({
      itemIndex,
      pageKey,
      splitIndex,
      stem,
      width,
      height,
      corners,
      rectRing,
      clipRing,
      scalePixelsPerFoot,
      rotationDegrees,
      gcps: points,
      transformationType: transformation?.type ?? 'polynomial',
    });
  });
  return pages;
}
