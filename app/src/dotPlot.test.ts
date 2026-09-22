import { describe, expect, it } from 'vitest';

import { domainOf, dotAt, layoutDots, scaleFor } from './dotPlot.ts';

describe('layoutDots', () => {
  it('puts well-separated values in the bottom row, at their own x', () => {
    const layout = layoutDots([0, 5, 10], 100, 40, 3);
    expect(layout.rows).toBe(1);
    expect(layout.dots.map((d) => d.row)).toEqual([0, 0, 0]);
    // x is the value's own position, not a bin's: the midpoint lands midway.
    expect(layout.dots[1]?.x).toBeCloseTo(50, 5);
    expect(layout.dots.map((d) => d.y)).toEqual([37, 37, 37]);
  });

  it('stacks values too close to sit side by side', () => {
    // Three identical values cannot share a row at any positive radius.
    const layout = layoutDots([1, 1, 1], 100, 40, 3);
    expect(layout.rows).toBe(3);
    expect(new Set(layout.dots.map((d) => d.row))).toEqual(new Set([0, 1, 2]));
    // ... and a single pile sits mid-axis rather than at the left edge.
    expect(layout.dots[0]?.x).toBeCloseTo(50, 5);
  });

  it('keeps the input order, whatever order the values arrive in', () => {
    const layout = layoutDots([10, 0, 5], 100, 40, 3);
    expect(layout.dots[0]?.x).toBeGreaterThan(layout.dots[2]?.x ?? 0);
    expect(layout.dots[2]?.x).toBeGreaterThan(layout.dots[1]?.x ?? 0);
  });

  it('squashes a pile too deep for the height instead of overflowing it', () => {
    // Chicago's rotations: 94 pages within a hair of each other, in 80px.
    const layout = layoutDots(new Array(94).fill(0.5), 200, 80, 3);
    expect(layout.rows).toBe(94);
    expect(layout.step).toBeLessThan(6);
    const ys = layout.dots.map((d) => d.y);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...ys)).toBe(77);
  });

  it('never squashes below a diameter when there is room', () => {
    const layout = layoutDots([1, 1, 1], 100, 400, 3);
    expect(layout.step).toBe(6);
  });

  it('has nothing to lay out for no values', () => {
    expect(layoutDots([], 100, 40, 3).dots).toEqual([]);
  });
});

describe('dotAt', () => {
  it('finds the dot under a pointer and nothing in empty space', () => {
    const layout = layoutDots([0, 5, 10], 100, 40, 3);
    const first = layout.dots[0];
    expect(first).toBeDefined();
    expect(dotAt(layout, first?.x ?? 0, first?.y ?? 0, 3)).toBe(0);
    expect(dotAt(layout, 50, 5, 3)).toBeNull();
  });

  it('picks the nearest when squashed dots overlap', () => {
    const layout = layoutDots(new Array(40).fill(2), 200, 60, 3);
    const target = layout.dots[10];
    expect(dotAt(layout, target?.x ?? 0, target?.y ?? 0, 3)).toBe(10);
  });
});

describe('scaleFor', () => {
  it('round-trips a value through pixels and back', () => {
    const scale = scaleFor([2.7, 6.1], 200, 3);
    for (const value of [2.7, 3.5, 4.4, 6.1]) {
      expect(scale.toValue(scale.toX(value))).toBeCloseTo(value, 6);
    }
  });

  it('insets the ends by the radius, so edge dots are not clipped', () => {
    const scale = scaleFor([0, 10], 100, 3);
    expect(scale.toX(0)).toBe(3);
    expect(scale.toX(10)).toBe(97);
  });

  it('agrees with the positions layoutDots produced', () => {
    // The drag maps pixels back to values; disagreeing by a pixel here brushes
    // a different point than the one under the cursor.
    const values = [2.74, 3.1, 5.6, 6.12];
    const layout = layoutDots(values, 200, 60, 3);
    const scale = scaleFor(layout.domain, 200, 3);
    layout.dots.forEach((dot, index) => {
      expect(scale.toValue(dot.x)).toBeCloseTo(values[index] ?? 0, 6);
    });
  });
});

describe('domainOf', () => {
  it('widens a degenerate range so the pile is not pinned to the left edge', () => {
    const [low, high] = domainOf([4, 4, 4]);
    expect(low).toBeLessThan(4);
    expect(high).toBeGreaterThan(4);
  });

  it('widens a run of zeros too, which has no magnitude to scale from', () => {
    expect(domainOf([0, 0])).toEqual([-0.5, 0.5]);
  });
});
