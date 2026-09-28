import { describe, expect, it } from 'vitest';

import {
  footprintFeatures,
  inMultiPolygon,
  newestVolumeAt,
  townsInView,
  yearOptions,
  type MultiPolygonCoords,
  type TownFootprints,
} from './footprints.ts';
import type { Volume } from './places.ts';

// A square footprint [x0, x1] x [y0, y1].
function square(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): MultiPolygonCoords {
  return [
    [
      [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0],
      ],
    ],
  ];
}

// A town with a whole-town 1890 atlas and two 1950 volumes, west and east.
const town: TownFootprints = {
  v1890: {
    year: 1890,
    anchor: [5, 5],
    footprint: square(0, 0, 10, 10),
    display: null,
  },
  west1950: {
    year: 1950,
    anchor: [2, 5],
    footprint: square(0, 0, 5, 10),
    display: true,
  },
  east1950: {
    year: 1950,
    anchor: [8, 5],
    footprint: square(5, 0, 10, 10),
    display: true,
  },
};

function volume(item: string, year: number | null, digitized = true): Volume {
  return {
    item,
    date: String(year),
    year,
    sheets: 10,
    title: 'T',
    ...(digitized ? { state: 's', mirrorYear: String(year) } : {}),
  };
}

describe('inMultiPolygon', () => {
  it('is inside the outer ring and outside its holes', () => {
    const donut: MultiPolygonCoords = [
      [...square(0, 0, 10, 10)[0]!, ...square(4, 4, 6, 6)[0]!],
    ];
    expect(inMultiPolygon(2, 2, donut)).toBe(true);
    expect(inMultiPolygon(5, 5, donut)).toBe(false);
    expect(inMultiPolygon(20, 5, donut)).toBe(false);
  });
});

describe('newestVolumeAt', () => {
  it('picks the newest volume covering the point', () => {
    const towns = new Map([['s/town', town]]);
    expect(newestVolumeAt(2, 5, towns)).toEqual({
      item: 'west1950',
      place: 's/town',
    });
    expect(newestVolumeAt(8, 5, towns)).toEqual({
      item: 'east1950',
      place: 's/town',
    });
    expect(newestVolumeAt(20, 5, towns)).toBeNull();
  });
});

describe('townsInView', () => {
  it('keeps the towns whose bounds meet the view', () => {
    const index = { 's/a': [0, 0, 1, 1], 's/b': [5, 5, 6, 6] } as const;
    expect(
      townsInView(
        index as unknown as Record<string, [number, number, number, number]>,
        [0.5, 0.5, 2, 2],
      ),
    ).toEqual(['s/a']);
  });
});

describe('yearOptions', () => {
  const catalogue = [
    volume('east1950', 1950),
    volume('west1950', 1950),
    volume('paper1920', 1920, false),
    volume('v1890', 1890),
  ];

  it('offers the volume covering the spot in each year', () => {
    expect(yearOptions([8, 5], 'east1950', catalogue, town)).toEqual([
      { year: 1950, item: 'east1950', status: 'placed', count: 1 },
      { year: 1920, item: null, status: 'not digitized', count: 1 },
      { year: 1890, item: 'v1890', status: 'placed', count: 1 },
    ]);
  });

  it('follows the spot from an old volume to the right new one', () => {
    const options = yearOptions([2, 5], 'v1890', catalogue, town);
    expect(options[0]).toEqual({
      year: 1950,
      item: 'west1950',
      status: 'placed',
      count: 1,
    });
  });

  it('leaves out a year whose volumes all cover somewhere else', () => {
    const elsewhere = {
      ...town,
      v1890: { ...town.v1890!, footprint: square(20, 20, 30, 30) },
    };
    const years = yearOptions([8, 5], 'east1950', catalogue, elsewhere).map(
      (o) => o.year,
    );
    expect(years).toEqual([1950, 1920]);
  });

  it('offers a digitized volume with no placed sheet as unplaced', () => {
    const withUnplaced = [...catalogue, volume('lost1905', 1905)];
    const options = yearOptions([8, 5], 'east1950', withUnplaced, town);
    expect(options.find((o) => o.year === 1905)).toEqual({
      year: 1905,
      item: 'lost1905',
      status: 'unplaced',
      count: 1,
    });
  });
});

describe('footprintFeatures', () => {
  it("draws each volume's display shape, and the selected one whole", () => {
    const towns = new Map([['s/town', town]]);
    const plain = footprintFeatures(towns, null);
    expect(plain.features.map((f) => f.id)).toEqual(['west1950', 'east1950']);
    const selected = footprintFeatures(towns, 'v1890');
    const old = selected.features.find((f) => f.id === 'v1890');
    expect(old?.properties?.selected).toBe(1);
    expect(old?.geometry).toEqual({
      type: 'MultiPolygon',
      coordinates: square(0, 0, 10, 10),
    });
  });
});
