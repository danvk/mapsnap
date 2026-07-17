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

import { projectThroughCorners } from '../geometry';
import type { PageGeo } from './pages';

const EARTH_RADIUS_FT = 20_925_524.0;

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

/** The 7×7 grid of pixel sample points `mapsnap compare` measures over. */
export function sampleGrid(width: number, height: number): [number, number][] {
  const points: [number, number][] = [];
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      points.push([(width * i) / 6, (height * j) / 6]);
    }
  }
  return points;
}

/** Fold a degree difference into (-180, 180]. */
function wrapDegrees(value: number): number {
  return ((value + 540) % 360) - 180;
}

/** Compare one generated page against its truth counterpart. */
export function comparePage(
  generated: PageGeo,
  truth: PageGeo,
): PageCompareStats {
  const { width, height } = generated;
  const distances = sampleGrid(width, height).map(([x, y]) =>
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

/**
 * Stats for every page of the loaded annotation, keyed by itemIndex.
 *
 * A page pairs with the truth item sharing its page key. Pages with no truth
 * counterpart are absent from the map; split panels (whose truth lives in the
 * parent canvas frame) map to null.
 */
export function compareToTruth(
  pages: PageGeo[],
  truthPages: PageGeo[],
): Map<number, PageCompareStats | null> {
  const truthByKey = new Map<string, PageGeo[]>();
  for (const truthPage of truthPages) {
    const list = truthByKey.get(truthPage.pageKey) ?? [];
    list.push(truthPage);
    truthByKey.set(truthPage.pageKey, list);
  }
  const stats = new Map<number, PageCompareStats | null>();
  for (const page of pages) {
    const isSplit = page.pageKey.includes('__');
    const parentKey = page.pageKey.split('__')[0] ?? page.pageKey;
    const candidates =
      truthByKey.get(page.pageKey) ?? truthByKey.get(parentKey);
    if (!candidates || candidates.length === 0) continue;
    if (isSplit || candidates.length > 1) {
      // Truth for a split is in the parent-canvas pixel frame; comparing
      // would need the panel offset from pN.panels.json.
      stats.set(page.itemIndex, null);
      continue;
    }
    stats.set(page.itemIndex, comparePage(page, candidates[0]!));
  }
  return stats;
}
