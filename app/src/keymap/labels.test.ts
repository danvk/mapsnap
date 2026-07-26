import { describe, expect, it } from 'vitest';
import { pointInPolygon } from '../geometry';
import {
  createLabelsJson,
  labelBox,
  DEFAULT_BOX_HEIGHT,
  DEFAULT_BOX_WIDTH,
} from './labels';

describe('createLabelsJson', () => {
  it('assembles the sidecar payload', () => {
    const labels = [{ x: 1, y: 2, text: '21' }];
    expect(createLabelsJson(100, 200, labels)).toEqual({
      width: 100,
      height: 200,
      labels,
    });
  });
});

describe('labelBox', () => {
  it('centers a box of the given size on the point', () => {
    const box = labelBox(100, 100, DEFAULT_BOX_WIDTH, DEFAULT_BOX_HEIGHT);
    const hw = DEFAULT_BOX_WIDTH / 2;
    const hh = DEFAULT_BOX_HEIGHT / 2;
    expect(box).toEqual([
      [100 - hw, 100 - hh],
      [100 + hw, 100 - hh],
      [100 + hw, 100 + hh],
      [100 - hw, 100 + hh],
    ]);
  });

  it('contains its own center but not far-away points', () => {
    const box = labelBox(100, 100, 40, 40);
    expect(pointInPolygon(100, 100, box)).toBe(true);
    expect(pointInPolygon(115, 105, box)).toBe(true);
    expect(pointInPolygon(200, 100, box)).toBe(false);
  });

  it('resizes with the slider values, so hit-testing tracks what is drawn', () => {
    const wide = labelBox(100, 100, 300, 40);
    expect(pointInPolygon(240, 100, wide)).toBe(true);
    const narrow = labelBox(100, 100, 40, 40);
    expect(pointInPolygon(240, 100, narrow)).toBe(false);
  });
});
