/**
 * Smoothed thin-plate spline, a port of `mapsnap.keymap.snap.thin_plate_spline`.
 *
 * The key-map model keymap-snap places pages in is a spline through the
 * sheet's inlier intersections with a large smoothing term (TPS_SMOOTHING),
 * which is what keeps straight streets straight between GCPs: on New Orleans
 * 1896 the exactly-interpolating spline bends a street's midpoint 9 ft off its
 * chord (median; 53 ft at p90) while the smoothed one bends it 1.3 ft (6.7 at
 * p90) at the same held-out accuracy. allmaps only interpolates exactly, so to
 * draw the pipeline's surface the debugger evaluates this spline at the GCP
 * pixels and hands allmaps the smoothed targets; the exact spline through
 * those reproduces the smoothed surface (measured: 0.0 ft apart).
 *
 * Same kernel, system and smoothing as the Python: U(r²) = ½ r² ln r², the
 * smoothing added to the kernel's diagonal, an affine part [1, x, y].
 */

export type Point2 = [number, number];

/** Mirrors `mapsnap.keymap.snap.TPS_SMOOTHING`. */
export const TPS_SMOOTHING = 1e6;

// ½ r² ln r² for r² > 0, and 0 at r = 0 (the kernel's removable singularity).
function kernel(squared: number): number {
  return squared > 0 ? squared * 0.5 * Math.log(squared) : 0;
}

/**
 * Solve A x = B for a square A by Gaussian elimination with partial pivoting.
 * B may have several columns; both are overwritten. Sized for a few hundred
 * GCPs (a 500x500 system is a few milliseconds).
 */
export function solveLinear(a: number[][], b: number[][]): number[][] {
  const n = a.length;
  const width = b[0]?.length ?? 0;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (a[pivot][col] === 0) throw new Error('singular spline system');
    if (pivot !== col) {
      [a[col], a[pivot]] = [a[pivot], a[col]];
      [b[col], b[pivot]] = [b[pivot], b[col]];
    }
    for (let row = col + 1; row < n; row++) {
      const factor = a[row][col] / a[col][col];
      if (factor === 0) continue;
      for (let k = col; k < n; k++) a[row][k] -= factor * a[col][k];
      for (let k = 0; k < width; k++) b[row][k] -= factor * b[col][k];
    }
  }
  const x: number[][] = Array.from({ length: n }, () => Array(width).fill(0));
  for (let row = n - 1; row >= 0; row--) {
    for (let k = 0; k < width; k++) {
      let sum = b[row][k];
      for (let col = row + 1; col < n; col++) sum -= a[row][col] * x[col][k];
      x[row][k] = sum / a[row][row];
    }
  }
  return x;
}

/**
 * The smoothed spline mapping `source` points onto `destination`, as a
 * function of query points. With smoothing 0 it interpolates exactly.
 */
export function thinPlateSpline(
  source: Point2[],
  destination: Point2[],
  smoothing: number = TPS_SMOOTHING,
): (queries: Point2[]) => Point2[] {
  const count = source.length;
  const size = count + 3;
  const system: number[][] = Array.from({ length: size }, () =>
    Array(size).fill(0),
  );
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < count; j++) {
      const dx = source[i][0] - source[j][0];
      const dy = source[i][1] - source[j][1];
      system[i][j] = kernel(dx * dx + dy * dy) + (i === j ? smoothing : 0);
    }
    const poly = [1, source[i][0], source[i][1]];
    for (let k = 0; k < 3; k++) {
      system[i][count + k] = poly[k];
      system[count + k][i] = poly[k];
    }
  }
  const rhs: number[][] = Array.from({ length: size }, (unused, i) =>
    i < count ? [destination[i][0], destination[i][1]] : [0, 0],
  );
  const solution = solveLinear(system, rhs);
  const weights = solution.slice(0, count);
  const affine = solution.slice(count);
  return (queries) =>
    queries.map(([x, y]) => {
      let outX = affine[0][0] + affine[1][0] * x + affine[2][0] * y;
      let outY = affine[0][1] + affine[1][1] * x + affine[2][1] * y;
      for (let i = 0; i < count; i++) {
        const dx = x - source[i][0];
        const dy = y - source[i][1];
        const value = kernel(dx * dx + dy * dy);
        outX += value * weights[i][0];
        outY += value * weights[i][1];
      }
      return [outX, outY];
    });
}
