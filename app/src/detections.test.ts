import { describe, expect, it } from 'vitest';
import { confidenceColor, filterDetections } from './detections';
import type { Detection } from './types';

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
