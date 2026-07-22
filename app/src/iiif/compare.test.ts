import { describe, expect, it } from 'vitest';

import { rmseBucket, statsByItemIndex } from './compare';
import type { ComparePageStats } from '../../server/compareTxt';
import type { PageGeo } from './pages';

// A compare-sidecar row with the given generated key and RMSE.
function row(genPageKey: string, rmseFt: number): ComparePageStats {
  return {
    genPageKey,
    rmseFt,
    maxFt: rmseFt * 1.5,
    translationFt: rmseFt * 0.8,
    rotationErrorDegrees: 0.5,
    scaleErrorPercent: 1.2,
  };
}

// A page with only the fields statsByItemIndex reads.
function page(stem: string, itemIndex: number): PageGeo {
  return { stem, itemIndex } as PageGeo;
}

describe('rmseBucket', () => {
  it('buckets RMSE by its boundaries', () => {
    expect(rmseBucket(25)).toBe('good');
    expect(rmseBucket(80)).toBe('ok');
    expect(rmseBucket(150)).toBe('poor');
    expect(rmseBucket(400)).toBe('disaster');
  });
});

describe('statsByItemIndex', () => {
  it('keys sidecar rows to pages by file stem, skipping unpaired pages', () => {
    const rows = [row('p1499l', 10.5), row('p1499n__2', 12.8)];
    const pages = [page('p1499l', 0), page('p1499n__2', 1), page('p1407', 2)];
    const stats = statsByItemIndex(rows, pages);
    expect(stats.get(0)?.rmseFt).toBe(10.5);
    expect(stats.get(1)?.rmseFt).toBe(12.8);
    expect(stats.has(2)).toBe(false); // no row for p1407
  });

  it('stores just the compare metrics (drops the generated key)', () => {
    const stats = statsByItemIndex([row('p1', 5)], [page('p1', 0)]);
    expect(stats.get(0)).toEqual({
      rmseFt: 5,
      maxFt: 7.5,
      translationFt: 4,
      rotationErrorDegrees: 0.5,
      scaleErrorPercent: 1.2,
    });
  });
});
