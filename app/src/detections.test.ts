import { describe, expect, it } from 'vitest';
import {
  confidenceColor,
  detectionFromAdjacency,
  FILL_YELLOW_HUE_BAND,
  filterDetections,
  isOnBuildingFill,
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

describe('detectionFromAdjacency', () => {
  const read50 = {
    number: 50,
    text: '50',
    confidence: 0.98,
    polygon: [
      [100, 200],
      [140, 200],
      [140, 245],
      [100, 245],
    ] as [number, number][],
    height: 45,
    x_frac: 0.9,
    y_frac: 0.5,
    edge: 'R',
    claim: true,
  };

  it('converts a digit read to Detection shape with bbox sides', () => {
    const det = detectionFromAdjacency(read50, new Set([50]));
    expect(det.text).toBe('50');
    expect(det.confidence).toBe(0.98);
    expect(det.long_side).toBe(45);
    expect(det.short_side).toBe(40);
    expect(det.ignore).toBe(false);
  });

  it('marks a claim of a reciprocated neighbor as mutual', () => {
    expect(detectionFromAdjacency(read50, new Set([50])).mutual).toBe(true);
    expect(detectionFromAdjacency(read50, new Set([51])).mutual).toBe(false);
  });

  it('marks non-claims as ignored, with no mutual flag', () => {
    const det = detectionFromAdjacency(
      {
        number: 2,
        text: '2',
        confidence: 0.9,
        polygon: [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10],
        ],
        height: 10,
        x_frac: 0.5,
        y_frac: 0.5,
        edge: 'center',
        claim: false,
      },
      new Set([2]),
    );
    expect(det.ignore).toBe(true);
    expect(det.mutual).toBeUndefined();
  });
});

describe('isOnBuildingFill', () => {
  const withBackground = (hue: number): Detection => ({
    polygon: [
      [0, 0],
      [10, 0],
      [10, 5],
      [0, 5],
    ],
    text: 'REP',
    confidence: 1,
    angle: 0,
    long_side: 10,
    short_side: 5,
    background: { color: '#c04040', hue, chroma: 12 },
  });

  it('is false when OCR recorded no background (the label is on paper)', () => {
    const { background: _background, ...onPaper } = withBackground(0);
    expect(isOnBuildingFill(onPaper)).toBe(false);
  });

  it('is true for the red brick and blue stone of the Sanborn colour code', () => {
    expect(isOnBuildingFill(withBackground(5))).toBe(true);
    expect(isOnBuildingFill(withBackground(250))).toBe(true);
  });

  it('is false inside the yellow/brown band, which paper and tape share', () => {
    expect(isOnBuildingFill(withBackground(93))).toBe(false);
    expect(isOnBuildingFill(withBackground(102.7))).toBe(false);
  });

  it('treats the band edges as spared', () => {
    const [low, high] = FILL_YELLOW_HUE_BAND;
    expect(isOnBuildingFill(withBackground(low))).toBe(false);
    expect(isOnBuildingFill(withBackground(high))).toBe(false);
    expect(isOnBuildingFill(withBackground(low - 0.1))).toBe(true);
  });
});
