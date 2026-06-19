import { describe, expect, it } from 'vitest';
import { parseDroppedJson } from './fileLoading';

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
});
