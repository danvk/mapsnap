import { describe, expect, it } from 'vitest';
import { pointInPolygon } from '../geometry';
import {
  createLabelsJson,
  labelBox,
  LABEL_BOX_HEIGHT,
  LABEL_BOX_WIDTH,
} from './labels';

describe('createLabelsJson', () => {
  it('assembles the sidecar payload', () => {
    const labels = [{ x: 1, y: 2, text: '21' }];
    expect(createLabelsJson('foo.jpg', 100, 200, labels)).toEqual({
      image: 'foo.jpg',
      width: 100,
      height: 200,
      labels,
    });
  });
});

describe('labelBox', () => {
  it('centers a box of the default size on the point', () => {
    const box = labelBox(100, 100);
    const hw = LABEL_BOX_WIDTH / 2;
    const hh = LABEL_BOX_HEIGHT / 2;
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
});
