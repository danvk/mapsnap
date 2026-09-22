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
    // -2 to -1 folded catches a page at -91.8 raw, which the same numbers read
    // unfolded would miss. Dropping `folded` would silently filter the wrong
    // pages rather than fail.
    const filter = {
      metric: 'rotation' as const,
      folded: true,
      range: [-2, -1] as [number, number],
    };
    expect(passesFilter(page(3, -1.8), filter)).toBe(true);
    expect(passesFilter(page(3, -91.8), filter)).toBe(true);
    expect(passesFilter(page(3, 40), filter)).toBe(false);
    expect(passesFilter(page(3, -91.8), { ...filter, folded: false })).toBe(
      false,
    );
  });
});

describe('the rotation fold', () => {
  it('lands a quarter-turned page on its upright siblings', () => {
    // Miami p11 sits at -91.6 degrees and the rest of the volume near -1.8;
    // folded they are the same grid, two tenths of a degree apart.
    expect(metricValue(rotationMetric, page(3, -1.8), true)).toBeCloseTo(-1.8);
    expect(metricValue(rotationMetric, page(3, -91.6), true)).toBeCloseTo(-1.6);
  });

  it('keeps a mode that straddles zero together', () => {
    // The reason the window is centered rather than [0, 90): these two pages
    // are eight tenths of a degree apart, and a fold starting at zero would put
    // them at opposite ends of the axis.
    const below = metricValue(rotationMetric, page(3, -0.4), true);
    const above = metricValue(rotationMetric, page(3, 0.4), true);
    expect(above - below).toBeCloseTo(0.8);
  });

  it('stays inside [-45, 45)', () => {
    for (const degrees of [-180, -90, -45, -0.001, 0, 44.999, 45, 179.5]) {
      const folded = metricValue(rotationMetric, page(3, degrees), true);
      expect(folded).toBeGreaterThanOrEqual(-45);
      expect(folded).toBeLessThan(45);
    }
  });

  it('leaves the value alone when not folded', () => {
    expect(metricValue(rotationMetric, page(3, -91.6), false)).toBe(-91.6);
  });

  it('marks upright on the rotation axis, and nothing on scale', () => {
    // Which side of upright a page leans is the thing to read off a rotation
    // chart; no Sanborn scale is anywhere near zero, so a zero there would only
    // ever fall off the end of the axis.
    expect(rotationMetric.origin).toBe(0);
    expect(METRICS[0]?.origin).toBeUndefined();
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
