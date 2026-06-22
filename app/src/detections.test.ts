import { describe, expect, it } from 'vitest';
import {
  confidenceColor,
  filterDetections,
  previewOrientation,
} from './detections';
import type { Detection } from './types';

/** Axis-aligned rectangle [width x height] polygon (clockwise from origin). */
function rect(width: number, height: number): [number, number][] {
  return [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
}

/** Parallelogram whose longest side points along `deg` (length `len`). */
function tilted(deg: number, len: number): [number, number][] {
  const r = (deg * Math.PI) / 180;
  const dx = Math.cos(r) * len;
  const dy = Math.sin(r) * len;
  // Perpendicular short offset so the long side is unambiguously longest.
  const ox = -Math.sin(r) * 10;
  const oy = Math.cos(r) * 10;
  return [
    [0, 0],
    [dx, dy],
    [dx + ox, dy + oy],
    [ox, oy],
  ];
}

describe('previewOrientation', () => {
  it('does not rotate an axis-aligned wide box (long side horizontal)', () => {
    const { textAngle, longHorizontal } = previewOrientation(rect(20, 10), 0);
    expect(textAngle).toBeCloseTo(0);
    expect(longHorizontal).toBe(true);
  });

  it('does not rotate an axis-aligned tall box (a number stays upright)', () => {
    const { textAngle, longHorizontal } = previewOrientation(rect(10, 20), 0);
    expect(textAngle).toBeCloseTo(0);
    expect(longHorizontal).toBe(false);
  });

  it('straightens a slightly tilted box the short way, long side horizontal', () => {
    const { textAngle, longHorizontal } = previewOrientation(
      tilted(10, 100),
      0,
    );
    expect(textAngle).toBeCloseTo((10 * Math.PI) / 180);
    expect(longHorizontal).toBe(true);
  });

  it('keeps a near-vertical box upright (short side horizontal)', () => {
    const { textAngle, longHorizontal } = previewOrientation(
      tilted(80, 100),
      0,
    );
    expect(textAngle).toBeCloseTo((-10 * Math.PI) / 180);
    expect(longHorizontal).toBe(false);
  });

  it('rotates a vertical street (angle 90) to horizontal', () => {
    const { textAngle, longHorizontal } = previewOrientation(rect(10, 20), 90);
    expect(textAngle).toBeCloseTo(Math.PI / 2);
    expect(longHorizontal).toBe(true);
  });

  it('rotates angle 270 the opposite way from angle 90', () => {
    const at90 = previewOrientation(rect(10, 20), 90);
    const at270 = previewOrientation(rect(10, 20), 270);
    expect(at270.textAngle).toBeCloseTo(-Math.PI / 2);
    expect(at90.textAngle).toBeCloseTo(-at270.textAngle); // 180° apart
  });
});

function makeDetection(overrides: Partial<Detection> = {}): Detection {
  return {
    polygon: [
      [0, 0],
      [10, 0],
      [10, 5],
      [0, 5],
    ],
    text: 'TEST',
    confidence: 0.5,
    angle: 0,
    long_side: 10,
    short_side: 5,
    ...overrides,
  };
}

describe('confidenceColor', () => {
  it('maps 0 to red and 1 to green', () => {
    expect(confidenceColor(0)).toBe('hsl(0, 90%, 45%)');
    expect(confidenceColor(1)).toBe('hsl(120, 90%, 45%)');
  });
});

describe('filterDetections', () => {
  const detections = [
    makeDetection({ confidence: 0.1, short_side: 5, long_side: 30 }),
    makeDetection({ confidence: 0.9, short_side: 25, long_side: 100 }),
    makeDetection({
      confidence: 0.9,
      short_side: 25,
      long_side: 100,
      ignore: true,
    }),
  ];

  it('keeps original indices', () => {
    const result = filterDetections(detections, {
      minConfidence: 0,
      minShortSide: 0,
      minLongSide: 0,
      showIgnored: true,
    });
    expect(result.map((r) => r.i)).toEqual([0, 1, 2]);
  });

  it('filters by confidence and side thresholds', () => {
    const result = filterDetections(detections, {
      minConfidence: 0.5,
      minShortSide: 20,
      minLongSide: 50,
      showIgnored: true,
    });
    expect(result.map((r) => r.i)).toEqual([1, 2]);
  });

  it('hides ignored detections unless showIgnored is set', () => {
    const hidden = filterDetections(detections, {
      minConfidence: 0,
      minShortSide: 0,
      minLongSide: 0,
      showIgnored: false,
    });
    expect(hidden.map((r) => r.i)).toEqual([0, 1]);
  });
});
