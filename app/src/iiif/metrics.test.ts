import { describe, expect, it } from 'vitest';

import { METRICS, metricValue, passesFilter } from './metrics.ts';
import type { PageGeo } from './pages.ts';

function page(scale: number, rotation: number): PageGeo {
  return {
    scalePixelsPerFoot: scale,
    rotationDegrees: rotation,
  } as PageGeo;
}

const rotationMetric = METRICS[1]!;

describe('passesFilter', () => {
  it('keeps everything when there is no filter', () => {
    expect(passesFilter(page(3, 0), null)).toBe(true);
  });

  it('is inclusive at both ends, so a brush to a dot keeps that dot', () => {
    const filter = {
      metric: 'scale' as const,
      folded: false,
      range: [2.7, 3.0] as [number, number],
    };
    expect(passesFilter(page(2.7, 0), filter)).toBe(true);
    expect(passesFilter(page(3.0, 0), filter)).toBe(true);
    expect(passesFilter(page(2.69, 0), filter)).toBe(false);
    expect(passesFilter(page(3.01, 0), filter)).toBe(false);
  });

  it('filters on rotation independently of scale', () => {
    const filter = {
      metric: 'rotation' as const,
      folded: false,
      range: [-1, 1] as [number, number],
    };
    expect(passesFilter(page(99, 0.5), filter)).toBe(true);
    expect(passesFilter(page(3, 45), filter)).toBe(false);
  });

  it('reads a folded filter on the folded axis', () => {
    // 87 to 89 on the folded axis is -3 to -1 raw, which the same numbers read
    // unfolded would exclude. Dropping `folded` would silently filter the wrong
    // pages rather than fail.
    const filter = {
      metric: 'rotation' as const,
      folded: true,
      range: [87, 89] as [number, number],
    };
    expect(passesFilter(page(3, -1.8), filter)).toBe(true);
    expect(passesFilter(page(3, -91.8), filter)).toBe(true);
    expect(passesFilter(page(3, 45), filter)).toBe(false);
    expect(passesFilter(page(3, -1.8), { ...filter, folded: false })).toBe(
      false,
    );
  });
});

describe('the rotation fold', () => {
  it('lands a quarter-turned page on its upright siblings', () => {
    // Miami p11 sits at -91.6 degrees and the rest of the volume near -1.8;
    // folded they are the same grid, two tenths of a degree apart.
    expect(metricValue(rotationMetric, page(3, -1.8), true)).toBeCloseTo(88.2);
    expect(metricValue(rotationMetric, page(3, -91.6), true)).toBeCloseTo(88.4);
  });

  it('stays inside [0, 90)', () => {
    for (const degrees of [-180, -90, -0.001, 0, 89.999, 90, 179.5]) {
      const folded = metricValue(rotationMetric, page(3, degrees), true);
      expect(folded).toBeGreaterThanOrEqual(0);
      expect(folded).toBeLessThan(90);
    }
  });

  it('leaves the value alone when not folded', () => {
    expect(metricValue(rotationMetric, page(3, -91.6), false)).toBe(-91.6);
  });

  it('offers no fold for scale, which has no period to fold on', () => {
    expect(METRICS[0]?.fold).toBeUndefined();
    expect(metricValue(METRICS[0]!, page(0.74, 0), true)).toBe(0.74);
  });
});

describe('METRICS', () => {
  it('charts scale and rotation, and nothing that is constant by construction', () => {
    // Skew is 0 and anisotropy 1 for any similarity, which every generated fit
    // is; charting them would be a column of identical dots.
    expect(METRICS.map((m) => m.key)).toEqual(['scale', 'rotation']);
  });

  it('reads each metric off a page', () => {
    const subject = page(4.25, -2.5);
    expect(METRICS[0]?.of(subject)).toBe(4.25);
    expect(METRICS[1]?.of(subject)).toBe(-2.5);
    expect(METRICS[0]?.format(4.256)).toBe('4.26');
    expect(METRICS[1]?.format(-2.54)).toBe('-2.5');
  });
});
