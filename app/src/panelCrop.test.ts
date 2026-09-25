import { describe, expect, it } from 'vitest';
import {
  panelCrop,
  panelIndexFromStem,
  parentStem,
  siblingPanelsPaths,
} from './panelCrop';

describe('panelIndexFromStem', () => {
  it('reads the 1-based index a split stem names', () => {
    expect(panelIndexFromStem('p20__3')).toBe(3);
    expect(panelIndexFromStem('p1__1')).toBe(1);
  });

  it('returns null for a stem that names no panel', () => {
    expect(panelIndexFromStem('p20')).toBeNull();
    expect(panelIndexFromStem('p0a')).toBeNull();
    // A key map's own stem can carry letters; only __N is a panel.
    expect(panelIndexFromStem('p1499H')).toBeNull();
  });
});

describe('parentStem', () => {
  it('strips the panel suffix', () => {
    expect(parentStem('p20__3')).toBe('p20');
    expect(parentStem('p20')).toBe('p20');
  });
});

describe('panelCrop', () => {
  const square: [number, number][] = [
    [10, 20],
    [110, 20],
    [110, 220],
    [10, 220],
  ];

  it('is the polygon bounding box', () => {
    const crop = panelCrop([square], 1, 500, 500);
    expect(crop).toEqual({
      x: 10,
      y: 20,
      width: 100,
      height: 200,
      ring: square,
    });
  });

  it('rounds the way write_panels does', () => {
    // floor on the near edges, round on the far ones: a detection that lands a
    // pixel out is invisible until it is not.
    const ragged: [number, number][] = [
      [10.7, 20.7],
      [110.6, 20.7],
      [110.6, 220.4],
      [10.7, 220.4],
    ];
    const crop = panelCrop([ragged], 1, 500, 500);
    expect(crop?.x).toBe(10);
    expect(crop?.y).toBe(20);
    expect(crop?.width).toBe(101); // round(110.6) - 10
    expect(crop?.height).toBe(200); // round(220.4) - 20
  });

  it('clamps to the image', () => {
    const over: [number, number][] = [
      [-5, -5],
      [600, -5],
      [600, 600],
      [-5, 600],
    ];
    const crop = panelCrop([over], 1, 500, 400);
    expect(crop).toMatchObject({ x: 0, y: 0, width: 500, height: 400 });
  });

  it('returns null for a missing or empty panel', () => {
    expect(panelCrop([square], 2, 500, 500)).toBeNull();
    expect(panelCrop([], 1, 500, 500)).toBeNull();
    expect(panelCrop([[]], 1, 500, 500)).toBeNull();
  });
});

describe('siblingPanelsPaths', () => {
  it("looks beside a mirror run's reads, then beside the sheet", () => {
    expect(
      siblingPanelsPaths(
        'data/wernersville_pa_1914/p2.jpg',
        'data/wernersville_pa_1914/runs/corpus-v1/p2__2.streets.json',
      ),
    ).toEqual([
      'data/wernersville_pa_1914/runs/corpus-v1/p2.panels.json',
      'data/wernersville_pa_1914/p2.panels.json',
    ]);
  });

  it('names one place when the image and reads share a directory', () => {
    expect(
      siblingPanelsPaths(
        'data/werner_pa_1914/p2.jpg',
        'data/werner_pa_1914/p2__2.streets.json',
      ),
    ).toEqual(['data/werner_pa_1914/p2.panels.json']);
  });

  it('is empty unless the JSON names a panel of that image', () => {
    expect(
      siblingPanelsPaths('data/v/p2.jpg', 'data/v/p2.streets.json'),
    ).toEqual([]);
    expect(
      siblingPanelsPaths('data/v/p3.jpg', 'data/v/p2__2.streets.json'),
    ).toEqual([]);
  });
});
