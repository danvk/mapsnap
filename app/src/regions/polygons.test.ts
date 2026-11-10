import { describe, expect, it } from 'vitest';
import {
  PREVIEW_MIN_MARGIN,
  REGION_COLORS,
  autoLabel,
  closedRing,
  createRegionsJson,
  openRing,
  previewCrop,
  rectRing,
  regionAt,
  regionColor,
  sheetFraction,
  vertexAt,
} from './polygons';
import type { KeymapDetection, RegionPolygon } from './types';

const SQUARE: [number, number][] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
];

function region(ring: [number, number][], text = ''): RegionPolygon {
  return { ring, text };
}

describe('closedRing', () => {
  it('repeats the first vertex so the ring is explicitly closed', () => {
    expect(closedRing(SQUARE)).toEqual([
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0],
    ]);
  });

  it('leaves an already-closed ring alone', () => {
    const closed: [number, number][] = [...SQUARE, [0, 0]];
    expect(closedRing(closed)).toHaveLength(5);
  });

  it('rounds to the one decimal the panels schema uses', () => {
    expect(closedRing([[1.234, 5.678]])[0]).toEqual([1.2, 5.7]);
  });

  it('handles an empty ring', () => {
    expect(closedRing([])).toEqual([]);
  });
});

describe('openRing', () => {
  it('round-trips with closedRing', () => {
    expect(openRing(closedRing(SQUARE))).toEqual(SQUARE);
  });

  it('leaves an open ring alone', () => {
    expect(
      openRing([
        [0, 0],
        [1, 1],
      ]),
    ).toEqual([
      [0, 0],
      [1, 1],
    ]);
  });
});

describe('createRegionsJson', () => {
  it('writes parallel panels and labels arrays', () => {
    const json = createRegionsJson(100, 200, [region(SQUARE, '12')]);
    expect(json).toEqual({
      width: 100,
      height: 200,
      panels: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10],
          [0, 0],
        ],
      ],
      labels: ['12'],
    });
  });
});

describe('autoLabel', () => {
  const detections: KeymapDetection[] = [
    { text: '12', x: 5, y: 5 },
    { text: '13', x: 50, y: 50 },
  ];

  it('names a ring that encloses exactly one page number', () => {
    // The whole point: a block contains its own printed number, so the user types nothing.
    expect(autoLabel(SQUARE, detections)).toBe('12');
  });

  it('stays blank when the ring encloses nothing', () => {
    expect(
      autoLabel(
        [
          [100, 100],
          [110, 100],
          [110, 110],
        ],
        detections,
      ),
    ).toBe('');
  });

  it('stays blank when the ring encloses two different numbers', () => {
    // Ambiguous: a ring spanning two blocks must not silently pick one.
    const wide: [number, number][] = [
      [0, 0],
      [60, 0],
      [60, 60],
      [0, 60],
    ];
    expect(autoLabel(wide, detections)).toBe('');
  });

  it('accepts a repeated detection of the same number', () => {
    const doubled = [...detections, { text: '12', x: 6, y: 6 }];
    expect(autoLabel(SQUARE, doubled)).toBe('12');
  });

  it('ignores a key another region already claimed', () => {
    expect(autoLabel(SQUARE, detections, new Set(['12']))).toBe('');
  });

  it('ignores blank detection text', () => {
    expect(autoLabel(SQUARE, [{ text: '', x: 5, y: 5 }])).toBe('');
  });

  it('needs a real ring', () => {
    expect(
      autoLabel(
        [
          [0, 0],
          [1, 1],
        ],
        detections,
      ),
    ).toBe('');
  });
});

describe('regionAt', () => {
  it('finds the region under a point', () => {
    expect(regionAt([region(SQUARE)], 5, 5)).toBe(0);
    expect(regionAt([region(SQUARE)], 50, 50)).toBeNull();
  });

  it('prefers the most recently drawn of two overlapping regions', () => {
    const bigger: [number, number][] = [
      [0, 0],
      [20, 0],
      [20, 20],
      [0, 20],
    ];
    expect(regionAt([region(SQUARE), region(bigger)], 5, 5)).toBe(1);
  });
});

describe('vertexAt', () => {
  it('finds a vertex within the radius and ignores one outside it', () => {
    expect(vertexAt(region(SQUARE), 10, 1, 3)).toBe(1);
    expect(vertexAt(region(SQUARE), 5, 5, 3)).toBeNull();
  });

  it('picks the nearest when two are in range', () => {
    expect(vertexAt(region(SQUARE), 9, 9, 100)).toBe(2);
  });
});

describe('sheetFraction', () => {
  it('reports the share of the sheet a ring covers', () => {
    expect(sheetFraction(SQUARE, 10, 10)).toBeCloseTo(1.0);
    expect(sheetFraction(SQUARE, 20, 20)).toBeCloseTo(0.25);
  });

  it('is zero for a degenerate ring or sheet', () => {
    expect(
      sheetFraction(
        [
          [0, 0],
          [1, 1],
        ],
        10,
        10,
      ),
    ).toBe(0);
    expect(sheetFraction(SQUARE, 0, 0)).toBe(0);
  });
});

describe('rectRing', () => {
  it('spans two opposite corners clockwise from the top left', () => {
    expect(rectRing([2, 3], [8, 9])).toEqual([
      [2, 3],
      [8, 3],
      [8, 9],
      [2, 9],
    ]);
  });

  it('normalizes a drag made in any direction', () => {
    // Dragging up-left must give the same rectangle as dragging down-right.
    const expected = rectRing([2, 3], [8, 9]);
    expect(rectRing([8, 9], [2, 3])).toEqual(expected);
    expect(rectRing([2, 9], [8, 3])).toEqual(expected);
    expect(rectRing([8, 3], [2, 9])).toEqual(expected);
  });

  it('produces a ring the rest of the pipeline accepts', () => {
    const ring = rectRing([0, 0], [10, 20]);
    expect(ring).toHaveLength(4);
    expect(sheetFraction(ring, 10, 20)).toBeCloseTo(1.0);
    expect(closedRing(ring)).toHaveLength(5);
  });
});

describe('previewCrop', () => {
  const sheet: [number, number] = [5000, 8000];
  const box: [number, number] = [320, 420];

  it('pads the bounding box by 15% of its larger side and shows it at 1:1', () => {
    const crop = previewCrop(rectRing([1000, 2000], [1200, 2100]), sheet, box)!;
    expect(crop.scale).toBe(1);
    expect(crop.x).toBeCloseTo(1000 - 30);
    expect(crop.y).toBeCloseTo(2000 - 30);
    expect(crop.width).toBeCloseTo(200 + 60);
    expect(crop.height).toBeCloseTo(100 + 60);
  });

  it('scales a block larger than the box down to fit', () => {
    const crop = previewCrop(rectRing([500, 500], [1500, 900]), sheet, box)!;
    expect(crop.width).toBeCloseTo(1000 + 300);
    expect(crop.width * crop.scale).toBeCloseTo(320);
    expect(crop.height * crop.scale).toBeLessThanOrEqual(420);
  });

  it('uses the minimum margin for a tiny block and stays inside the sheet', () => {
    const crop = previewCrop(rectRing([0, 0], [50, 50]), sheet, box)!;
    expect(crop.x).toBe(0);
    expect(crop.y).toBe(0);
    expect(crop.width).toBe(50 + 2 * PREVIEW_MIN_MARGIN);
    expect(crop.scale).toBe(1);
  });

  it('clips the window to a sheet smaller than the padded box', () => {
    const crop = previewCrop(rectRing([10, 10], [90, 90]), [100, 100], box)!;
    expect(crop).toMatchObject({ x: 0, y: 0, width: 100, height: 100 });
  });

  it('is null for an empty ring', () => {
    expect(previewCrop([], sheet, box)).toBeNull();
  });
});

describe('regionColor', () => {
  it('cycles through the palette by index', () => {
    expect(regionColor(0)).toBe(REGION_COLORS[0]);
    expect(regionColor(REGION_COLORS.length)).toBe(REGION_COLORS[0]);
    expect(regionColor(1)).not.toBe(regionColor(0));
  });
});
