import type { Corners, Street } from './types';

/** Solve the 3x3 linear system Ax = b using Gaussian elimination with partial pivoting. */
export function solveLinear3(A: number[][], b: number[]): number[] | null {
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[maxRow][col])) maxRow = row;
    }
    [m[col], m[maxRow]] = [m[maxRow], m[col]];
    if (Math.abs(m[col][col]) < 1e-12) return null;
    for (let row = col + 1; row < 3; row++) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k <= 3; k++) m[row][k] -= factor * m[col][k];
    }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    x[i] = m[i][3];
    for (let j = i + 1; j < 3; j++) x[i] -= m[i][j] * x[j];
    x[i] /= m[i][i];
  }
  return x;
}

/** Ray-casting point-in-polygon test for a convex or concave polygon. */
export function pointInPolygon(
  x: number,
  y: number,
  polygon: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Area of a polygon via the shoelace formula. Winding order does not matter. */
export function polygonArea(polygon: [number, number][]): number {
  let sum = 0;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    sum += xj * yi - xi * yj;
  }
  return Math.abs(sum) / 2;
}

/** Haversine distance in miles between two lon/lat points. */
export function distanceMiles(
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number,
): number {
  const R = 3958.8;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

/**
 * Fit affine transform lon = a0 + a1*x + a2*y (and lat) from GCPs with lat/lon,
 * then map the 4 image corners to geographic coordinates.
 *
 * Returns [nw, ne, se, sw] in [lon, lat] order, or null if fewer than 3 GCPs.
 */
export function computeCorners(
  gcps: Street[],
  width: number,
  height: number,
): Corners | null {
  const valid = gcps.filter((p) => p.lat !== undefined && p.lon !== undefined);
  if (valid.length < 3) return null;

  const AtA = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const AtbLon = [0, 0, 0];
  const AtbLat = [0, 0, 0];
  for (const p of valid) {
    const row = [1, p.x, p.y];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) AtA[i][j] += row[i] * row[j];
      AtbLon[i] += row[i] * p.lon!;
      AtbLat[i] += row[i] * p.lat!;
    }
  }

  const lonCoeff = solveLinear3(AtA, AtbLon);
  const latCoeff = solveLinear3(AtA, AtbLat);
  if (!lonCoeff || !latCoeff) return null;

  const transform = (x: number, y: number): [number, number] => [
    lonCoeff[0] + lonCoeff[1] * x + lonCoeff[2] * y,
    latCoeff[0] + latCoeff[1] * x + latCoeff[2] * y,
  ];

  return [
    transform(0, 0),
    transform(width, 0),
    transform(width, height),
    transform(0, height),
  ];
}

/**
 * A closed ring of [lon, lat] points approximating a circle of `radiusMeters`
 * around (lon, lat), using an equirectangular local approximation. Suitable for
 * drawing the key-map neighborhood on a web map at city scale.
 */
export function circlePolygon(
  lon: number,
  lat: number,
  radiusMeters: number,
  points = 64,
): [number, number][] {
  const dLat = radiusMeters / 110540;
  const dLon = radiusMeters / (111320 * Math.cos((lat * Math.PI) / 180));
  const ring: [number, number][] = [];
  for (let i = 0; i <= points; i++) {
    const theta = (i / points) * 2 * Math.PI;
    ring.push([lon + dLon * Math.cos(theta), lat + dLat * Math.sin(theta)]);
  }
  return ring;
}
