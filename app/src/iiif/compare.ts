/**
 * Truth-comparison types and display buckets for the volume viewer.
 *
 * The per-page error metrics come from the pipeline's `mapsnap compare` sidecar table (parsed
 * by server/compareTxt, fetched via fetchCompare) rather than being recomputed in the browser,
 * so split-panel associations and RMSEs match exactly what `mapsnap compare` reports. This
 * module only shapes those rows for display and buckets RMSE into colours.
 */

import type { ComparePageStats } from '../../server/compareTxt';
import type { PageGeo } from './pages';

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
  /** Shear angle in degrees, when the table reports it; else undefined. */
  skewDegrees?: number;
  /** Anisotropy (x/y scale ratio, 1 = isotropic), when reported; else undefined. */
  anisotropy?: number;
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

/**
 * Per-page compare stats keyed by itemIndex, pairing each sidecar row to the generated page
 * whose file stem matches its generated key.
 *
 * The sidecar already resolved split-panel associations, so pages that were split on either
 * side are keyed correctly here. A page with no paired row (no truth counterpart) is simply
 * absent, so the map and table leave it uncoloured.
 */
export function statsByItemIndex(
  rows: ComparePageStats[],
  pages: PageGeo[],
): Map<number, PageCompareStats> {
  const byKey = new Map(rows.map((row) => [row.genPageKey, row]));
  const stats = new Map<number, PageCompareStats>();
  for (const page of pages) {
    const row = byKey.get(page.stem);
    if (!row) continue;
    stats.set(page.itemIndex, {
      rmseFt: row.rmseFt,
      maxFt: row.maxFt,
      translationFt: row.translationFt,
      rotationErrorDegrees: row.rotationErrorDegrees,
      scaleErrorPercent: row.scaleErrorPercent,
      skewDegrees: row.skewDegrees,
      anisotropy: row.anisotropy,
    });
  }
  return stats;
}
