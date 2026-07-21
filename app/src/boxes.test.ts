import { describe, expect, it } from 'vitest';
import {
  boxArea,
  boxAngleColor,
  boxesFromJson,
  boxSideLengths,
  filterBoxes,
  unrotatePoint,
} from './boxes';
import type { Box, BoxesJsonData } from './types';

describe('unrotatePoint', () => {
  it('is the identity at angle 0', () => {
    expect(unrotatePoint(63, 31, 0, 1409, 2037)).toEqual([63, 31]);
  });

  it('inverts the 90° (CCW) rotation', () => {
    expect(unrotatePoint(1071, -1, 90, 1409, 2037)).toEqual([1409, 1071]);
    expect(unrotatePoint(1107, 15, 90, 1409, 2037)).toEqual([1393, 1107]);
  });

  it('inverts the 270° (CW) rotation', () => {
    expect(unrotatePoint(1235, 5, 270, 1409, 2037)).toEqual([5, 801]);
    expect(unrotatePoint(1267, 19, 270, 1409, 2037)).toEqual([19, 769]);
  });
});

describe('boxesFromJson', () => {
  const data: BoxesJsonData = {
    width: 1409,
    height: 2037,
    timestamp: 'now',
    command: [],
    boxes: [
      {
        angle: 0,
        horizontal_list: [[63, 93, 31, 49]],
        free_list: [
          [
            [867, 44],
            [900, 35],
            [904, 53],
            [871, 62],
          ],
        ],
      },
      {
        angle: 90,
        horizontal_list: [[1071, 1107, -1, 15]],
        free_list: [],
      },
      {
        angle: 270,
        horizontal_list: [[1235, 1267, 5, 19]],
        free_list: [],
      },
    ],
  };

  it('maps every box into the original image frame, tagged by angle', () => {
    const boxes = boxesFromJson(data);
    // Two angle-0 boxes (one horizontal, one free) plus one each at 90 and 270.
    expect(boxes.map((b) => b.angle)).toEqual([0, 0, 90, 270]);
  });

  it('turns an axis-aligned box into its four corners at angle 0', () => {
    const [box] = boxesFromJson(data);
    expect(box.polygon).toEqual([
      [63, 31],
      [93, 31],
      [93, 49],
      [63, 49],
    ]);
  });

  it('records each box long/short side lengths', () => {
    const [box] = boxesFromJson(data);
    // 30px wide, 18px tall.
    expect(box.long_side).toBe(30);
    expect(box.short_side).toBe(18);
  });

  it('leaves free-list polygons untouched at angle 0', () => {
    expect(boxesFromJson(data)[1].polygon).toEqual([
      [867, 44],
      [900, 35],
      [904, 53],
      [871, 62],
    ]);
  });

  it('unrotates 90° and 270° boxes back onto the image', () => {
    const boxes = boxesFromJson(data);
    expect(boxes[2].polygon).toEqual([
      [1409, 1071],
      [1409, 1107],
      [1393, 1107],
      [1393, 1071],
    ]);
    expect(boxes[3].polygon).toEqual([
      [5, 801],
      [5, 769],
      [19, 769],
      [19, 801],
    ]);
  });
});

describe('boxSideLengths', () => {
  it('averages opposite edges of a skewed quad', () => {
    const { long, short } = boxSideLengths([
      [0, 0],
      [10, 0],
      [10, 4],
      [0, 4],
    ]);
    expect(long).toBe(10);
    expect(short).toBe(4);
  });
});

function makeBox(long: number, short: number, angle = 0): Box {
  return { polygon: [], angle, long_side: long, short_side: short };
}

describe('filterBoxes', () => {
  const boxes = [makeBox(30, 18), makeBox(200, 40), makeBox(10, 8)];

  it('keeps boxes meeting both side minimums, with original indices', () => {
    const kept = filterBoxes(boxes, { minShortSide: 15, minLongSide: 20 });
    expect(kept.map((b) => b.i)).toEqual([0, 1]);
  });

  it('drops a box failing either minimum', () => {
    expect(
      filterBoxes(boxes, { minShortSide: 20, minLongSide: 0 }),
    ).toHaveLength(1);
  });
});

describe('boxArea', () => {
  it('multiplies the two side lengths', () => {
    expect(boxArea(makeBox(30, 18))).toBe(540);
  });
});

describe('boxAngleColor', () => {
  it('gives each standard rotation a distinct color', () => {
    const colors = [0, 90, 270].map(boxAngleColor);
    expect(new Set(colors).size).toBe(3);
  });
});
