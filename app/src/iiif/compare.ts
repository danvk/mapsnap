/**
 * Per-page truth comparison for the volume viewer, mirroring `mapsnap compare`.
 *
 * Both the loaded annotation and the volume's truth (main.iiif.json) are
 * rewritten into the same local-image pixel frame by the server, and each
 * page's corner quad reproduces its fitted transform exactly (an affine maps
 * the corner rectangle to a parallelogram, and bilinear interpolation through
 * a parallelogram is that affine). So the compare metrics reduce to mapping a
 * 7×7 pixel grid through both corner quads and measuring the separation —
 * the same measurement `mapsnap compare` reports.
 *
 * Split panels are the exception: truth expresses a split in the parent
 * page's canvas while our panels are their own images, and relating the two
 * frames needs pN.panels.json. Those pages get `null` stats rather than a
 * wrong number.
 */

import { pointInPolygon, projectThroughCorners } from '../geometry';
import type { PageGeo } from './pages';

const EARTH_RADIUS_FT = 20_925_524.0;

/** Minimum panel overlap (IoU) to treat a truth and generated split as the same region. */
const MIN_SPLIT_IOU = 0.1;

/** Grid resolution for the sampled polygon-IoU used to match split panels. */
const IOU_GRID = 40;

/** Compare metrics for one page, matching `mapsnap compare`'s columns. */
export interface PageCompareStats {
  rmseFt: number;
  maxFt: number;
  /** Error at the image center: the translation component. */
  translationFt: number;
  /** Signed rotation difference in degrees. */
  rotationErrorDegrees: number;
  /** Scale difference in percent (generated relative to truth). */
  scaleErrorPercent: number;
}

/** RMSE quality bucket, shared by the page table and the map color-coding. */
export type RmseBucket = 'good' | 'ok' | 'poor' | 'disaster';

/** Bucket boundaries: <=25 good, <=100 ok, <=200 poor, beyond is a disaster. */
export function rmseBucket(rmseFt: number): RmseBucket {
  if (rmseFt <= 25) return 'good';
  if (rmseFt <= 100) return 'ok';
  if (rmseFt <= 200) return 'poor';
  return 'disaster';
}

/** Display color per bucket (also used by the .rmse-* table classes). */
export const RMSE_BUCKET_COLORS: Record<RmseBucket, string> = {
  good: '#1a7f37',
  ok: '#9a6700',
  poor: '#e8590c',
  disaster: '#cf222e',
};

/** Great-circle distance in feet between two [lon, lat] points. */
export function haversineFeet(
  a: [number, number],
  b: [number, number],
): number {
  const toRadians = (degrees: number) => (degrees * Math.PI) / 180;
  const [lon1, lat1] = a.map(toRadians) as [number, number];
  const [lon2, lat2] = b.map(toRadians) as [number, number];
  const sinLat = Math.sin((lat2 - lat1) / 2);
  const sinLon = Math.sin((lon2 - lon1) / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return 2 * EARTH_RADIUS_FT * Math.asin(Math.sqrt(h));
}

/** A pixel-frame rectangle to sample over: [minX, minY, maxX, maxY]. */
export type SampleRegion = [number, number, number, number];

/** A 7×7 grid of pixel sample points spanning the given region. */
export function sampleGridRegion(region: SampleRegion): [number, number][] {
  const [minX, minY, maxX, maxY] = region;
  const points: [number, number][] = [];
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      points.push([
        minX + ((maxX - minX) * i) / 6,
        minY + ((maxY - minY) * j) / 6,
      ]);
    }
  }
  return points;
}

/** The 7×7 grid of pixel sample points `mapsnap compare` measures over. */
export function sampleGrid(width: number, height: number): [number, number][] {
  return sampleGridRegion([0, 0, width, height]);
}

/** Fold a degree difference into (-180, 180]. */
function wrapDegrees(value: number): number {
  return ((value + 540) % 360) - 180;
}

/**
 * Compare one generated page against its truth counterpart.
 *
 * `region` restricts the error grid to a sub-rectangle of the image (both pages share the
 * same local pixel frame), so a split panel is measured only over its own pixels rather than
 * the whole sheet — mirroring `mapsnap compare`'s split-canvas sampling. Defaults to the
 * whole image.
 */
export function comparePage(
  generated: PageGeo,
  truth: PageGeo,
  region?: SampleRegion,
): PageCompareStats {
  const { width, height } = generated;
  const distances = sampleGridRegion(region ?? [0, 0, width, height]).map(
    ([x, y]) =>
      haversineFeet(
        projectThroughCorners(generated.corners, width, height, x, y),
        projectThroughCorners(truth.corners, truth.width, truth.height, x, y),
      ),
  );
  const rmseFt = Math.sqrt(
    distances.reduce((sum, d) => sum + d * d, 0) / distances.length,
  );
  const translationFt = haversineFeet(
    projectThroughCorners(
      generated.corners,
      width,
      height,
      width / 2,
      height / 2,
    ),
    projectThroughCorners(
      truth.corners,
      truth.width,
      truth.height,
      width / 2,
      height / 2,
    ),
  );
  return {
    rmseFt,
    maxFt: Math.max(...distances),
    translationFt,
    rotationErrorDegrees: wrapDegrees(
      generated.rotationDegrees - truth.rotationDegrees,
    ),
    scaleErrorPercent:
      (truth.scalePixelsPerFoot / generated.scalePixelsPerFoot - 1) * 100,
  };
}

// Bounding box [minX, minY, maxX, maxY] of a pixel-frame polygon.
function polygonBounds(polygon: [number, number][]): SampleRegion {
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

// A boolean IOU_GRID×IOU_GRID rasterization of a local-frame polygon over [0,w]×[0,h].
function polygonMask(
  polygon: [number, number][],
  width: number,
  height: number,
): boolean[] {
  const mask: boolean[] = new Array(IOU_GRID * IOU_GRID);
  for (let i = 0; i < IOU_GRID; i++) {
    const x = ((i + 0.5) / IOU_GRID) * width;
    for (let j = 0; j < IOU_GRID; j++) {
      const y = ((j + 0.5) / IOU_GRID) * height;
      mask[i * IOU_GRID + j] = pointInPolygon(x, y, polygon);
    }
  }
  return mask;
}

// Intersection-over-union of two rasterized masks.
function maskIou(a: boolean[], b: boolean[]): number {
  let inter = 0;
  let union = 0;
  for (let k = 0; k < a.length; k++) {
    if (a[k] && b[k]) inter++;
    if (a[k] || b[k]) union++;
  }
  return union > 0 ? inter / union : 0;
}

/**
 * Pair generated and truth split panels by greatest panel overlap, returning generated
 * itemIndex → matched truth page. Mirrors `match_split_pairs` in compare_iiif_georef.py:
 * every pair with IoU ≥ MIN_SPLIT_IOU is considered and the highest-overlap pairs are
 * assigned first, each panel used once. A generated page we kept whole (no split index)
 * stands in with the full page rectangle, so it pairs with the truth split it most overlaps
 * — the largest one. All polygons are in the shared local pixel frame.
 */
function matchSplitPairs(
  genPages: PageGeo[],
  truthPages: PageGeo[],
): Map<number, PageGeo> {
  const { width, height } = truthPages[0]!;
  const fullRect: [number, number][] = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  const genMasks = genPages.map((g) =>
    polygonMask(g.splitIndex != null ? g.clipPolygon : fullRect, width, height),
  );
  const truthMasks = truthPages.map((t) =>
    polygonMask(t.clipPolygon, width, height),
  );
  const candidates: { iou: number; gi: number; ti: number }[] = [];
  for (let gi = 0; gi < genPages.length; gi++) {
    for (let ti = 0; ti < truthPages.length; ti++) {
      const iou = maskIou(genMasks[gi]!, truthMasks[ti]!);
      if (iou >= MIN_SPLIT_IOU) candidates.push({ iou, gi, ti });
    }
  }
  candidates.sort((a, b) => b.iou - a.iou);
  const usedGen = new Set<number>();
  const usedTruth = new Set<number>();
  const pairs = new Map<number, PageGeo>();
  for (const { gi, ti } of candidates) {
    if (usedGen.has(gi) || usedTruth.has(ti)) continue;
    usedGen.add(gi);
    usedTruth.add(ti);
    pairs.set(genPages[gi]!.itemIndex, truthPages[ti]!);
  }
  return pairs;
}

// Group pages by their (parent) page key.
function groupByPageKey(pages: PageGeo[]): Map<string, PageGeo[]> {
  const groups = new Map<string, PageGeo[]>();
  for (const page of pages) {
    const list = groups.get(page.pageKey) ?? [];
    list.push(page);
    groups.set(page.pageKey, list);
  }
  return groups;
}

/**
 * Stats for every page of the loaded annotation, keyed by itemIndex.
 *
 * Pages with no truth counterpart are absent from the map. When a page is split on either
 * side — including the common case where we kept a page whole but the truth split it — the
 * panels are matched by overlap ({@link matchSplitPairs}) and each pair compared over the
 * generated panel's own region (the whole sheet for a page we kept whole). A generated
 * panel that no truth split overlaps maps to null.
 */
export function compareToTruth(
  pages: PageGeo[],
  truthPages: PageGeo[],
): Map<number, PageCompareStats | null> {
  const truthByKey = groupByPageKey(truthPages);
  const stats = new Map<number, PageCompareStats | null>();
  for (const [pageKey, genGroup] of groupByPageKey(pages)) {
    const truthGroup = truthByKey.get(pageKey);
    if (!truthGroup || truthGroup.length === 0) continue; // no truth for this page
    if (genGroup.length === 1 && truthGroup.length === 1) {
      stats.set(
        genGroup[0]!.itemIndex,
        comparePage(genGroup[0]!, truthGroup[0]!),
      );
      continue;
    }
    const pairs = matchSplitPairs(genGroup, truthGroup);
    for (const gen of genGroup) {
      const truth = pairs.get(gen.itemIndex);
      if (!truth) {
        stats.set(gen.itemIndex, null); // no overlapping truth panel
        continue;
      }
      const region =
        gen.splitIndex != null ? polygonBounds(gen.clipPolygon) : undefined;
      stats.set(gen.itemIndex, comparePage(gen, truth, region));
    }
  }
  return stats;
}
