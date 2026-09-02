import { describe, expect, it } from 'vitest';
import {
  confidenceColor,
  detectionFromAdjacency,
  FILL_YELLOW_HUE_BAND,
  filterDetections,
  isOnBuildingFill,
  matchesTextQuery,
  relaxedShortSide,
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
      minAspectRatio: 0,
      relaxation: false,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    });
    expect(result.map((r) => r.i)).toEqual([0, 1, 2]);
  });

  it('filters by confidence and side thresholds', () => {
    const result = filterDetections(detections, {
      minConfidence: 0.5,
      minShortSide: 20,
      minLongSide: 50,
      minAspectRatio: 0,
      relaxation: false,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    });
    expect(result.map((r) => r.i)).toEqual([1, 2]);
  });

  it('keeps inset-masked reads visible whatever the ignored toggle says', () => {
    // A masked read is the thing to look at on a key map (#276); it is not `ignore`.
    const dets = [
      makeDetection({ text: '311', inset: true }),
      makeDetection({ text: '310' }),
    ];
    const shown = filterDetections(dets, {
      minConfidence: 0,
      minShortSide: 0,
      minLongSide: 0,
      minAspectRatio: 0,
      relaxation: false,
      highConfidenceSizeFraction: 0.7,
      showIgnored: false,
      text: '',
    });
    expect(shown.map((r) => r.det.text)).toEqual(['311', '310']);
  });

  it('hides ignored detections unless showIgnored is set', () => {
    const hidden = filterDetections(detections, {
      minConfidence: 0,
      minShortSide: 0,
      minLongSide: 0,
      minAspectRatio: 0,
      relaxation: false,
      highConfidenceSizeFraction: 0.7,
      showIgnored: false,
      text: '',
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

describe('matchesTextQuery', () => {
  it('matches everything when the query is empty or blank', () => {
    expect(matchesTextQuery('ELLIOTT ST', '')).toBe(true);
    expect(matchesTextQuery('ELLIOTT ST', '   ')).toBe(true);
  });

  it('matches a page number without matching numbers that contain it', () => {
    // The motivating case: finding page 6 among a key map's 201 reads, where a
    // substring match would also return 16, 60 and 63.
    expect(matchesTextQuery('6', '6')).toBe(true);
    expect(matchesTextQuery('16', '6')).toBe(false);
    expect(matchesTextQuery('60', '6')).toBe(false);
    expect(matchesTextQuery('63', '6')).toBe(false);
  });

  it('treats a letter suffix as part of the page key', () => {
    // 6S is a different sheet from 6.
    expect(matchesTextQuery('6S', '6')).toBe(false);
    expect(matchesTextQuery('6S', '6S')).toBe(true);
  });

  it('matches a word inside a longer street name', () => {
    expect(matchesTextQuery('ELLIOTT ST', 'ELLIOTT')).toBe(true);
    expect(matchesTextQuery('ELLIOTT ST', 'ST')).toBe(true);
    expect(matchesTextQuery('ELLIOTTS', 'ELLIOTT')).toBe(false);
  });

  it('ignores case', () => {
    expect(matchesTextQuery('ELLIOTT ST', 'elliott')).toBe(true);
  });

  it('ORs comma-separated terms and ignores the spacing', () => {
    expect(matchesTextQuery('7', '6, 7')).toBe(true);
    expect(matchesTextQuery('6', '6,7')).toBe(true);
    expect(matchesTextQuery('8', '6, 7')).toBe(false);
    // Trailing separators and stray blanks must not turn into a match-nothing term.
    expect(matchesTextQuery('8', '6, ,')).toBe(false);
  });

  it('treats regex metacharacters literally', () => {
    expect(matchesTextQuery('6', '.')).toBe(false);
    expect(matchesTextQuery('N. MAIN', '.')).toBe(true);
  });
});

describe('relaxedShortSide', () => {
  it('returns the full floor at or below min confidence', () => {
    expect(relaxedShortSide(0.15, 0.15, 20, 0.7)).toBe(20);
    expect(relaxedShortSide(0.1, 0.15, 20, 0.7)).toBe(20);
  });

  it('returns the high-confidence floor at confidence 1', () => {
    expect(relaxedShortSide(1, 0.15, 20, 0.7)).toBeCloseTo(14, 6);
  });

  it('interpolates monotonically between the two', () => {
    const mid = relaxedShortSide(0.5, 0.15, 20, 0.7);
    expect(mid).toBeLessThan(20);
    expect(mid).toBeGreaterThan(14);
  });

  it('is disabled when the fraction would not lower the floor', () => {
    expect(relaxedShortSide(1, 0.15, 20, 1)).toBe(20);
  });
});

describe('filterDetections aspect ratio and relaxation', () => {
  const wide = makeDetection({
    text: 'BROADWAY',
    confidence: 0.9,
    short_side: 10,
    long_side: 90,
  });
  const squat = makeDetection({
    text: 'ELM',
    confidence: 0.9,
    short_side: 40,
    long_side: 50,
  });

  it('rejects a detection below the aspect-ratio floor', () => {
    const base = {
      minConfidence: 0,
      minShortSide: 0,
      minLongSide: 0,
      relaxation: false,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    };
    expect(
      filterDetections([wide, squat], { ...base, minAspectRatio: 0 }),
    ).toHaveLength(2);
    expect(
      filterDetections([wide, squat], { ...base, minAspectRatio: 2 }),
    ).toHaveLength(1);
  });

  it('admits an undersized but confident detection only with relaxation on', () => {
    const small = makeDetection({
      text: 'OAK',
      confidence: 1,
      short_side: 15,
      long_side: 40,
    });
    const base = {
      minConfidence: 0.15,
      minShortSide: 20,
      minLongSide: 40,
      minAspectRatio: 0,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    };
    expect(
      filterDetections([small], { ...base, relaxation: false }),
    ).toHaveLength(0);
    const relaxed = filterDetections([small], { ...base, relaxation: true });
    expect(relaxed).toHaveLength(1);
    expect(relaxed[0].relaxed).toBe(true);
  });

  it('does not mark a detection that clears the strict floors', () => {
    const big = makeDetection({
      text: 'MAIN',
      confidence: 1,
      short_side: 30,
      long_side: 90,
    });
    const result = filterDetections([big], {
      minConfidence: 0.15,
      minShortSide: 20,
      minLongSide: 40,
      minAspectRatio: 0,
      relaxation: true,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    });
    expect(result).toHaveLength(1);
    expect(result[0].relaxed).toBeUndefined();
  });

  it('relaxes the long side in the same proportion as the short side', () => {
    // georef scales min_long_side by required_short/min_short; a detection just
    // under both floors must clear both relaxed floors, not only the short one.
    const shortOnly = makeDetection({
      text: 'PINE',
      confidence: 1,
      short_side: 15,
      long_side: 39,
    });
    const both = makeDetection({
      text: 'CEDAR',
      confidence: 1,
      short_side: 15,
      long_side: 30,
    });
    const base = {
      minConfidence: 0.15,
      minShortSide: 20,
      minLongSide: 40,
      minAspectRatio: 0,
      relaxation: true,
      highConfidenceSizeFraction: 0.7,
      showIgnored: true,
      text: '',
    };
    expect(filterDetections([shortOnly], base)).toHaveLength(1);
    expect(filterDetections([both], base)).toHaveLength(1);
  });
});
