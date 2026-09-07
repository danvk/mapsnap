import { describe, expect, it } from 'vitest';
import {
  solveLinear,
  thinPlateSpline,
  type Point2,
} from './thinPlateSpline.ts';

// Eight noisy GCPs and three query points, with the Python
// mapsnap.keymap.snap.thin_plate_spline evaluated on them (rng seed 7).
const SRC: Point2[] = [
  [3125.5, 4486.1],
  [3878.4, 1126.0],
  [1500.8, 4367.8],
  [26.3, 4106.1],
  [3985.3, 2339.7],
  [1515.2, 1392.1],
  [1274.3, 2225.4],
  [2522.7, 2767.5],
];
const DST: Point2[] = [
  [1927.75, -2921.73],
  [2555.77, -649.13],
  [765.34, -2909.17],
  [-221.88, -2774.56],
  [2705.53, -1402.95],
  [853.95, -903.19],
  [800.08, -1442.76],
  [1563.38, -1784.43],
];
const Q: Point2[] = [
  [1200.0, 2500.0],
  [4000.0, 800.0],
  [2600.0, 4100.0],
];

function close(actual: Point2[], expected: number[][], digits: number) {
  actual.forEach((point, i) => {
    expect(point[0]).toBeCloseTo(expected[i][0], digits);
    expect(point[1]).toBeCloseTo(expected[i][1], digits);
  });
}

// The spline system is badly scaled (kernel entries ~1e8 against a [1, x, y]
// block), so numpy reports it as numerically singular (cond ~1e17). The
// pipeline solves it with lstsq, which truncates the near-null direction;
// this port solves it exactly, matching numpy.linalg.solve to every digit.
// The two agree at the source points and differ elsewhere by up to ~2.3
// units on this eight-point toy (numpy's own solve-vs-lstsq gap); on New
// Orleans 1896's 485 GCPs the route's targets sit 0.65 ft median / 19 ft max
// from the pipeline's model, against a ~50 ft model error.
const LSTSQ_TOLERANCE = 3.0;
function near(actual: Point2[], expected: number[][]) {
  actual.forEach((point, i) => {
    expect(Math.abs(point[0] - expected[i][0])).toBeLessThan(LSTSQ_TOLERANCE);
    expect(Math.abs(point[1] - expected[i][1])).toBeLessThan(LSTSQ_TOLERANCE);
  });
}

describe('thinPlateSpline', () => {
  it('matches numpy.linalg.solve when barely smoothed, and interpolates', () => {
    const spline = thinPlateSpline(SRC, DST, 1e-9);
    close(
      spline(Q),
      [
        [746.79303, -1629.39961],
        [2632.83709, -435.08416],
        [1559.90509, -2684.59378],
      ],
      2,
    );
    // The pipeline's lstsq answer is within the truncation tolerance.
    near(spline(Q), [
      [747.1522, -1629.1677],
      [2630.566, -436.5506],
      [1561.2887, -2683.7004],
    ]);
    // Interpolates its own points.
    close(spline(SRC), DST, 3);
  });

  it('matches numpy.linalg.solve at the pipeline smoothing', () => {
    const spline = thinPlateSpline(SRC, DST, 1e6);
    close(
      spline(Q),
      [
        [703.86394, -1638.28134],
        [2647.63965, -425.65723],
        [1569.485, -2680.94141],
      ],
      2,
    );
    near(spline(Q), [
      [704.6595, -1637.6081],
      [2645.4303, -427.5268],
      [1570.5464, -2680.0432],
    ]);
    // Smoothing trades interpolation for a flatter surface: the GCPs no
    // longer pass through exactly.
    const [x] = spline([SRC[0]])[0];
    expect(Math.abs(x - DST[0][0])).toBeGreaterThan(1);
  });

  it('reproduces an affine at any smoothing (it is in the bending null space)', () => {
    const affine = (p: Point2): Point2 => [
      3 + 0.7 * p[0] - 0.1 * p[1],
      -8 - 0.05 * p[0] - 0.6 * p[1],
    ];
    for (const smoothing of [0, 1e6]) {
      const spline = thinPlateSpline(SRC, SRC.map(affine), smoothing);
      close(spline(Q), Q.map(affine), 4);
    }
  });
});

describe('solveLinear', () => {
  it('solves a small system with pivoting', () => {
    const x = solveLinear(
      [
        [0, 2, 1],
        [1, 1, 1],
        [2, 0, 3],
      ],
      [[5], [6], [13]],
    );
    // x = [2, 1, 3]
    expect(x.map((row) => row[0])).toEqual(
      [2, 1, 3].map((v) => expect.closeTo(v, 9)),
    );
  });
});
