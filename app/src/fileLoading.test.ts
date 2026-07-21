import { describe, expect, it } from 'vitest';
import { pageStem, parseDroppedJson } from './fileLoading';

const fallback = { width: 800, height: 600 };

describe('parseDroppedJson', () => {
  it('returns invalid for non-JSON text', () => {
    expect(parseDroppedJson('not json', fallback)).toEqual({ kind: 'invalid' });
  });

  it('treats a bare detection array as old-format streets', () => {
    const text = JSON.stringify([
      {
        polygon: [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ],
        text: 'A',
        confidence: 0.8,
        angle: 0,
        long_side: 1,
        short_side: 1,
      },
    ]);
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('streets');
    if (result.kind === 'streets') {
      expect(result.detections).toHaveLength(1);
      expect(result.width).toBe(800);
      expect(result.height).toBe(600);
    }
  });

  it('drops detections with non-positive confidence', () => {
    const text = JSON.stringify([
      {
        text: 'A',
        confidence: 0.8,
        polygon: [],
        angle: 0,
        long_side: 1,
        short_side: 1,
      },
      {
        text: 'B',
        confidence: 0,
        polygon: [],
        angle: 0,
        long_side: 1,
        short_side: 1,
      },
    ]);
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('streets');
    if (result.kind === 'streets') {
      expect(result.detections.map((d) => d.text)).toEqual(['A']);
    }
  });

  it('reads dimensions from new-format streets.json', () => {
    const text = JSON.stringify({
      width: 1234,
      height: 567,
      timestamp: 'now',
      command: [],
      streets: [
        {
          text: 'A',
          confidence: 0.8,
          polygon: [],
          angle: 0,
          long_side: 1,
          short_side: 1,
        },
      ],
    });
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('streets');
    if (result.kind === 'streets') {
      expect(result.width).toBe(1234);
      expect(result.height).toBe(567);
    }
  });

  it('recognizes panels.json and reads its dimensions', () => {
    const text = JSON.stringify({
      image: 'p20.raw.jpg',
      width: 1417,
      height: 2038,
      manual: true,
      panels: [
        [
          [0, 0],
          [100, 0],
          [100, 100],
          [0, 100],
        ],
        [
          [100, 0],
          [200, 0],
          [200, 200],
          [100, 200],
        ],
      ],
    });
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('panels');
    if (result.kind === 'panels') {
      expect(result.panels).toHaveLength(2);
      expect(result.width).toBe(1417);
      expect(result.height).toBe(2038);
      expect(result.labels).toBeUndefined();
    }
  });

  it('passes through panel labels when present', () => {
    const text = JSON.stringify({
      image: 'p0b.jpg',
      width: 5866,
      height: 7323,
      panels: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      ],
      labels: ['55'],
    });
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('panels');
    if (result.kind === 'panels') {
      expect(result.labels).toEqual(['55']);
    }
  });

  it('recognizes boxes.json and maps boxes into the image frame', () => {
    const text = JSON.stringify({
      width: 1409,
      height: 2037,
      timestamp: 'now',
      command: [],
      boxes: [
        { angle: 0, horizontal_list: [[63, 93, 31, 49]], free_list: [] },
        { angle: 90, horizontal_list: [[1071, 1107, -1, 15]], free_list: [] },
      ],
    });
    const result = parseDroppedJson(text, fallback);
    expect(result.kind).toBe('boxes');
    if (result.kind === 'boxes') {
      expect(result.width).toBe(1409);
      expect(result.height).toBe(2037);
      expect(result.boxes.map((b) => b.angle)).toEqual([0, 90]);
      // The 90° box is unrotated back onto the original image frame.
      expect(result.boxes[1].polygon[0]).toEqual([1409, 1071]);
    }
  });

  it('treats georef objects as georef', () => {
    const text = JSON.stringify({
      width: 100,
      height: 100,
      streets: [{ street: 'Main St', x: 1, y: 2 }],
      intersections: [],
    });
    const result = parseDroppedJson(text, fallback);
    expect(result).toEqual({ kind: 'georef', text });
  });

  it('classifies adjacency.json by its pages + adjacency keys', () => {
    const data = {
      pages: { p49: { number: 49, detections: [] } },
      adjacency: [['p49', 'p50']],
    };
    const result = parseDroppedJson(JSON.stringify(data), fallback);
    expect(result.kind).toBe('adjacency');
    if (result.kind === 'adjacency') {
      expect(result.data.pages.p49.number).toBe(49);
      expect(result.data.adjacency).toEqual([['p49', 'p50']]);
    }
  });
});

describe('pageStem', () => {
  it('strips directories and all extensions', () => {
    expect(pageStem('p49.jpg')).toBe('p49');
    expect(pageStem('data/vol/p50n.2048px.jpg')).toBe('p50n');
    expect(pageStem('/abs/path/p1b.jpg')).toBe('p1b');
  });

  it('matches split-panel stems', () => {
    expect(pageStem('p86__2.jpg')).toBe('p86__2');
  });
});
