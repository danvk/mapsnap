import { describe, expect, it } from 'vitest';
import {
  computeCorners,
  distanceMiles,
  pointInPolygon,
  solveLinear3,
} from './geometry';
import type { Street } from './types';

describe('solveLinear3', () => {
  it('solves a simple system', () => {
    // x = 1, y = 2, z = 3 for the identity-ish system below.
    const A = [
      [2, 0, 0],
      [0, 3, 0],
      [0, 0, 4],
    ];
    const b = [2, 6, 12];
    const x = solveLinear3(A, b);
    expect(x).not.toBeNull();
    expect(x![0]).toBeCloseTo(1);
    expect(x![1]).toBeCloseTo(2);
    expect(x![2]).toBeCloseTo(3);
  });

  it('solves a system that requires pivoting', () => {
    const A = [
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ];
    const b = [3, 4, 5];
    const x = solveLinear3(A, b);
    expect(x).not.toBeNull();
    // x + y = 5, x + z = 4, y + z = 3 → x=3, y=2, z=1
    expect(x![0]).toBeCloseTo(3);
    expect(x![1]).toBeCloseTo(2);
    expect(x![2]).toBeCloseTo(1);
  });

  it('returns null for a singular matrix', () => {
    const A = [
      [1, 2, 3],
      [2, 4, 6],
      [1, 1, 1],
    ];
    expect(solveLinear3(A, [1, 2, 3])).toBeNull();
  });
});

describe('pointInPolygon', () => {
  const square: [number, number][] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ];

  it('detects points inside', () => {
    expect(pointInPolygon(5, 5, square)).toBe(true);
  });

  it('detects points outside', () => {
    expect(pointInPolygon(15, 5, square)).toBe(false);
    expect(pointInPolygon(-1, -1, square)).toBe(false);
  });
});

describe('distanceMiles', () => {
  it('is zero for identical points', () => {
    expect(distanceMiles(-73.99, 40.7, -73.99, 40.7)).toBeCloseTo(0);
  });

  it('computes a known distance (~1 degree latitude ≈ 69 miles)', () => {
    expect(distanceMiles(0, 40, 0, 41)).toBeGreaterThan(68);
    expect(distanceMiles(0, 40, 0, 41)).toBeLessThan(70);
  });
});

describe('computeCorners', () => {
  it('returns null with fewer than 3 GCPs', () => {
    const streets: Street[] = [
      { street: 'A', x: 0, y: 0, lon: -74, lat: 40 },
      { street: 'B', x: 10, y: 0, lon: -73, lat: 40 },
    ];
    expect(computeCorners(streets, 10, 10)).toBeNull();
  });

  it('recovers an affine transform from exact GCPs', () => {
    // lon = -74 + 0.1*x, lat = 40 - 0.1*y
    const streets: Street[] = [
      { street: 'A', x: 0, y: 0, lon: -74, lat: 40 },
      { street: 'B', x: 100, y: 0, lon: -64, lat: 40 },
      { street: 'C', x: 0, y: 100, lon: -74, lat: 30 },
    ];
    const corners = computeCorners(streets, 100, 100);
    expect(corners).not.toBeNull();
    const [nw, ne, se, sw] = corners!;
    expect(nw[0]).toBeCloseTo(-74);
    expect(nw[1]).toBeCloseTo(40);
    expect(ne[0]).toBeCloseTo(-64);
    expect(se[1]).toBeCloseTo(30);
    expect(sw[0]).toBeCloseTo(-74);
  });
});
