import { describe, expect, it } from 'vitest';

import { METRICS, passesFilter } from './metrics.ts';
import type { PageGeo } from './pages.ts';

function page(scale: number, rotation: number): PageGeo {
  return {
    scalePixelsPerFoot: scale,
    rotationDegrees: rotation,
  } as PageGeo;
}

describe('passesFilter', () => {
  it('keeps everything when there is no filter', () => {
    expect(passesFilter(page(3, 0), null)).toBe(true);
  });

  it('is inclusive at both ends, so a brush to a dot keeps that dot', () => {
    const filter = {
      metric: 'scale' as const,
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
      range: [-1, 1] as [number, number],
    };
    expect(passesFilter(page(99, 0.5), filter)).toBe(true);
    expect(passesFilter(page(3, 45), filter)).toBe(false);
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
