import { describe, expect, it } from 'vitest';
import {
  circlePolygon,
  computeCorners,
  crosshairLines,
  distanceMiles,
  pointInPolygon,
  polygonArea,
  solveLinear3,
} from './geometry';
import type { Street } from './types';

describe('crosshairLines', () => {
  it('is a horizontal and a vertical segment centered on the point', () => {
    const [lon, lat, arm] = [-77.0, 38.9, 100];
    const [horiz, vert] = crosshairLines(lon, lat, arm);
    // Both segments pass through (lon, lat) at their midpoint.
    expect((horiz[0][1] + horiz[1][1]) / 2).toBeCloseTo(lat);
    expect((vert[0][0] + vert[1][0]) / 2).toBeCloseTo(lon);
    // Horizontal varies in lon only; vertical in lat only.
    expect(horiz[0][1]).toBeCloseTo(horiz[1][1]);
    expect(vert[0][0]).toBeCloseTo(vert[1][0]);
    // Each arm is ~arm metres from the center.
    expect(
      distanceMiles(lon, lat, horiz[1][0], horiz[1][1]) * 1609.34,
    ).toBeCloseTo(arm, -1);
  });
});

describe('circlePolygon', () => {
  it('is a closed ring whose points are ~radiusMeters from the center', () => {
    const [lon, lat, radius] = [-87.64, 41.89, 300];
    const ring = circlePolygon(lon, lat, radius, 32);
    expect(ring.length).toBe(33); // points + 1 (closed)
    expect(ring[0]).toEqual(ring[ring.length - 1]);
    // Every vertex is ~300 m from the center (allow a few % for the local approx).
    for (const [plon, plat] of ring) {
      const meters = distanceMiles(lon, lat, plon, plat) * 1609.34;
      expect(meters).toBeGreaterThan(radius * 0.95);
      expect(meters).toBeLessThan(radius * 1.05);
    }
  });
});

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

describe('polygonArea', () => {
  it('computes the area of a rectangle', () => {
    expect(
      polygonArea([
        [0, 0],
        [10, 0],
        [10, 5],
        [0, 5],
      ]),
    ).toBe(50);
  });

  it('is independent of winding order', () => {
    const cw: [number, number][] = [
      [0, 0],
      [0, 5],
      [10, 5],
      [10, 0],
    ];
    expect(polygonArea(cw)).toBe(50);
  });

  it('computes the area of a triangle', () => {
    expect(
      polygonArea([
        [0, 0],
        [4, 0],
        [0, 3],
      ]),
    ).toBe(6);
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
