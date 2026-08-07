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
import { splitIndexFor } from '../../server/iiifAnnotations';
import {
  computeCorners,
  pointInPolygon,
  projectThroughCorners,
} from '../geometry';
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
  /**
   * Deviation of the image x/y axes from perpendicular, in degrees.
   *
   * 0 for a similarity (our own 4-parameter fits are 0 by construction), so a
   * non-zero value means the annotation carries a transform that shears the
   * sheet — which a photographed flat map cannot physically do. Matches
   * `truth_distortion` in mapsnap/compare_iiif_georef.py, so it reads the same
   * as the `skew°` column of `mapsnap compare`.
   */
  skewDegrees: number;
  /**
   * x-scale / y-scale ratio. 1.0 for a similarity; > 1 means an image pixel
   * covers more ground horizontally than vertically. Same definition as the
   * `aniso` column of `mapsnap compare`.
   */
  anisotropy: number;
  gcps: PageGcp[];
  /** The annotation's transformation type, e.g. "polynomial" or "helmert". */
  transformationType: string;
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

/**
 * Skew (degrees off perpendicular) and anisotropy (x/y scale ratio) of a page.
 *
 * Port of `truth_distortion` in mapsnap/compare_iiif_georef.py, working from
 * the projected corners rather than the affine matrix: the metric per-pixel
 * step vectors are (ne − nw)/width and (sw − nw)/height, which are the affine's
 * two columns whenever the transform is affine, and its first-order behaviour
 * over the sheet otherwise.
 *
 * A similarity gives skew 0 and anisotropy 1. Truth annotations that deviate
 * far are usually bad reference data — a flat sheet photographed square cannot
 * shear — so the numbers belong next to scale and rotation in the info panel.
 */
export function distortion(
  corners: Corners,
  width: number,
  height: number,
): { skewDegrees: number; anisotropy: number } {
  const [nw, ne, , sw] = corners;
  const across = deltaMeters(nw, ne);
  const down = deltaMeters(nw, sw);
  const vx: [number, number] = [across[0] / width, across[1] / width];
  const vy: [number, number] = [down[0] / height, down[1] / height];
  const scaleX = Math.hypot(vx[0], vx[1]);
  const scaleY = Math.hypot(vy[0], vy[1]);
  if (scaleX === 0 || scaleY === 0) return { skewDegrees: 0, anisotropy: 1 };
  const cosAngle = Math.min(
    1,
    Math.max(-1, (vx[0] * vy[0] + vx[1] * vy[1]) / (scaleX * scaleY)),
  );
  return {
    skewDegrees: (Math.acos(cosAngle) * 180) / Math.PI - 90,
    anisotropy: scaleX / scaleY,
  };
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
 * Truth pages the loaded run never placed, ready to show as "missing" rows and
 * footprints.
 *
 * `missingKeys` comes from the compare sidecar's `(no fit)` rows, so this list is the
 * one the "N/M pages georeferenced" line counts — including split panels, which is
 * what the previous parent-key rule hid: a page whose sibling panel was fitted showed
 * no miss at all (Champaign reported four while showing none), and several missing
 * panels of one sheet collapsed into a single parent row (Hudson's p92__1, the largest
 * single miss in the corpus, was invisible).
 *
 * Negative `itemIndex` selection ids keep these clear of the fitted pages' array indices.
 */
export function missingTruthPages(
  truthPages: PageGeo[],
  missingKeys: readonly string[] | undefined,
): PageGeo[] {
  // A compare sidecar written before this field existed, or a server still serving
  // the older response, simply reports no misses rather than taking the page down.
  const wanted = new Set((missingKeys ?? []).map((key) => key.toLowerCase()));
  const missing: PageGeo[] = [];
  for (const truthPage of truthPages) {
    if (!wanted.has(truthPage.stem.toLowerCase())) continue;
    missing.push({ ...truthPage, itemIndex: -(missing.length + 1) });
  }
  return missing;
}

/** Whether a missing-page row knows where on earth its page belongs. */
export function hasFootprint(page: PageGeo): boolean {
  return page.clipRing.length >= 3 || page.rectRing.length >= 3;
}

/**
 * Un-fit pages of a volume that has no truth annotation, from the stems on disk.
 *
 * With truth, a miss is a truth page the run failed to place and its footprint is
 * known. Without truth there is nothing to compare against, so a miss is simply a
 * page image the annotation never placed — which is still worth listing, since it is
 * the only signal a truth-less volume gives about what is not working. These rows
 * carry no geometry (see {@link hasFootprint}): they appear in the page list, but
 * there is nowhere on the map to draw them.
 *
 * Negative `itemIndex` selection ids keep these clear of the fitted pages' array
 * indices, matching {@link missingTruthPages}.
 */
export function unfittedPages(
  fitPages: PageGeo[],
  volumeStems: readonly string[],
): PageGeo[] {
  const placed = new Set(fitPages.map((page) => page.stem.toLowerCase()));
  const missing: PageGeo[] = [];
  for (const stem of volumeStems) {
    if (placed.has(stem.toLowerCase())) continue;
    const [pageKey = stem, panel] = stem.split('__');
    missing.push({
      itemIndex: -(missing.length + 1),
      pageKey,
      splitIndex: panel === undefined ? null : Number(panel),
      stem,
      width: 0,
      height: 0,
      corners: [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
      ],
      rectRing: [],
      clipRing: [],
      scalePixelsPerFoot: 0,
      rotationDegrees: 0,
      skewDegrees: 0,
      anisotropy: 1,
      gcps: [],
      transformationType: '',
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
    // The server's `page` metadata is the on-disk stem, so for a split panel it
    // already ends in __N. PageGeo.pageKey is the PARENT key that panels share,
    // so strip the suffix here rather than appending a second one below --
    // otherwise a truth panel becomes "p844__3__3" and matches nothing.
    const stemFromMetadata = item.metadata?.find(
      (entry) => entry.label === 'page',
    )?.value;
    const pageKey = stemFromMetadata?.replace(/__\d+$/, '');
    if (!source?.width || !source.height || !stemFromMetadata || !pageKey)
      return;
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
    const stem =
      splitIndex != null ? `${pageKey}__${splitIndex}` : stemFromMetadata;

    const [nw, ne, , sw] = corners;
    const feetAcross = Math.hypot(...deltaMeters(nw, ne)) * FEET_PER_METER;
    const feetDown = Math.hypot(...deltaMeters(nw, sw)) * FEET_PER_METER;
    if (feetAcross === 0 || feetDown === 0) return;
    const scalePixelsPerFoot = (width / feetAcross + height / feetDown) / 2;
    const bearing = bearingDegrees(nw, ne);
    const rotationDegrees = ((bearing - 90 + 540) % 360) - 180;
    const { skewDegrees, anisotropy } = distortion(corners, width, height);

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
      skewDegrees,
      anisotropy,
      gcps: points,
      transformationType: transformation?.type ?? 'polynomial',
    });
  });
  return pages;
}
