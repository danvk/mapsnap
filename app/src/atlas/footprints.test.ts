import { describe, expect, it } from 'vitest';

import {
  defaultEdition,
  editionOptions,
  footprintFeatures,
  groupLabel,
  groupOfItem,
  inMultiPolygon,
  seriesAt,
  townsInView,
  volumeGroups,
  yearSpan,
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

function volume(
  item: string,
  year: number | null,
  options: { digitized?: boolean; number?: string } = {},
): Volume {
  return {
    item,
    date: String(year),
    year,
    sheets: 10,
    title: 'T',
    ...(options.number ? { volume: options.number } : {}),
    ...((options.digitized ?? true)
      ? { state: 's', mirrorYear: String(year) }
      : {}),
  };
}

// A town whose whole-town atlases of 1890 and 1900 were later split into
// vol. 1 (west) and vol. 2 (east), with a 1920 vol. 2 that nothing places and
// a 1925 supplement of additions only on paper.
const town: TownFootprints = {
  volumes: {
    t1890: { year: 1890, anchor: [5, 5], footprint: square(0, 0, 10, 10) },
    t1900: { year: 1900, anchor: [5, 5], footprint: square(0, 0, 10, 10) },
    west1950: { year: 1950, anchor: [2, 5], footprint: square(0, 0, 5, 10) },
    east1950: { year: 1950, anchor: [8, 5], footprint: square(5, 0, 10, 10) },
  },
  series: [
    {
      volume: '1',
      items: ['west1950'],
      footprint: square(0, 0, 5, 10),
      anchor: [2, 5],
      display: true,
    },
    {
      volume: '2',
      items: ['east1950', 'east1920'],
      footprint: square(5, 0, 10, 10),
      anchor: [8, 5],
      display: true,
    },
    {
      items: ['t1890', 't1900'],
      footprint: square(0, 0, 10, 10),
      anchor: [5, 5],
      display: null,
    },
  ],
};

const townVolumes = [
  volume('t1890', 1890),
  volume('t1900', 1900),
  volume('east1920', 1920, { number: '2' }),
  volume('supplement1925', 1925, { digitized: false }),
  volume('west1950', 1950, { number: '1' }),
  volume('east1950', 1950, { number: '2' }),
];

describe('inMultiPolygon', () => {
  it('is inside the outer ring and outside its holes', () => {
    const withHole: MultiPolygonCoords = [
      [...square(0, 0, 10, 10)[0]!, ...square(4, 4, 6, 6)[0]!],
    ];
    expect(inMultiPolygon(2, 2, withHole)).toBe(true);
    expect(inMultiPolygon(5, 5, withHole)).toBe(false);
    expect(inMultiPolygon(20, 5, withHole)).toBe(false);
  });
});

describe('townsInView', () => {
  it('keeps the towns whose bounds meet the view', () => {
    const towns = {
      'a/near': [0, 0, 1, 1] as [number, number, number, number],
      'a/far': [50, 50, 51, 51] as [number, number, number, number],
    };
    expect(townsInView(towns, [0.5, 0.5, 2, 2])).toEqual(['a/near']);
  });
});

describe('volumeGroups', () => {
  const groups = volumeGroups('s/town', townVolumes, town);

  it('makes a group of each series, editions newest first', () => {
    expect(groups.map((g) => g.editions.map((v) => v.item))).toEqual([
      ['west1950'],
      ['east1950', 'east1920'],
      ['t1900', 't1890'],
      ['supplement1925'],
    ]);
    expect(groups.map((g) => g.number)).toEqual(['1', '2', null, null]);
  });

  it('puts the editions no series holds in a last group', () => {
    expect(groups[3]).toMatchObject({ kind: 'rest', index: 3 });
  });

  it('makes one group of a town with nothing placed', () => {
    const bare = volumeGroups('s/town', townVolumes, null);
    expect(bare).toHaveLength(1);
    expect(bare[0]!.editions).toHaveLength(townVolumes.length);
  });

  it('finds the group of an edition', () => {
    expect(groupOfItem(groups, 'east1920')?.index).toBe(1);
    expect(groupOfItem(groups, 'nowhere')).toBeNull();
  });
});

describe('groupLabel', () => {
  const groups = volumeGroups('s/town', townVolumes, town);

  it('names a numbered volume by number, and the others by town and years', () => {
    expect(groups.map((g) => groupLabel(g, 'Town', groups.length))).toEqual([
      'Vol. 1',
      'Vol. 2',
      'Town, 1890–1900',
      'Other volumes',
    ]);
  });

  it("is the town's name when the town is one volume", () => {
    const only = volumeGroups('s/town', [volume('t1890', 1890)], null);
    expect(groupLabel(only[0]!, 'Town', 1)).toBe('Town');
  });
});

describe('yearSpan', () => {
  it('spans the years, oldest to newest', () => {
    expect(yearSpan([volume('a', 1950), volume('b', 1890)])).toBe('1890–1950');
    expect(yearSpan([volume('a', 1950)])).toBe('1950');
    expect(yearSpan([volume('a', null)])).toBe('undated');
  });
});

describe('seriesAt', () => {
  const towns = new Map([['s/town', town]]);

  it('lists the volumes covering a point, smallest first', () => {
    expect(seriesAt(2, 5, towns)).toEqual([
      { place: 's/town', index: 0 },
      { place: 's/town', index: 2 },
    ]);
  });

  it('never finds an overview atlas', () => {
    const overview: TownFootprints = {
      ...town,
      series: [
        ...town.series,
        {
          items: ['borough'],
          overview: true,
          footprint: square(1, 1, 3, 3),
          anchor: [2, 2],
          display: null,
        },
      ],
    };
    const hits = seriesAt(2, 2, new Map([['s/town', overview]]));
    expect(hits.map((hit) => hit.index)).toEqual([0, 2]);
  });

  it('is empty off every volume', () => {
    expect(seriesAt(50, 50, towns)).toEqual([]);
  });
});

describe('defaultEdition', () => {
  const editions = [
    volume('east1950', 1950),
    volume('east1920', 1920),
    volume('t1900', 1900),
  ];

  it('opens on the newest edition covering the point', () => {
    expect(defaultEdition(editions, town.volumes, [2, 5])?.item).toBe('t1900');
  });

  it('else on the newest placed edition', () => {
    expect(defaultEdition(editions, town.volumes, null)?.item).toBe('east1950');
  });

  it('else on the newest digitized one', () => {
    const unplaced = [
      volume('paper', 1930, { digitized: false }),
      volume('scan', 1920),
    ];
    expect(defaultEdition(unplaced, {}, null)?.item).toBe('scan');
  });
});

describe('editionOptions', () => {
  it('labels each edition by year, and says what there is of it', () => {
    const options = editionOptions(
      [
        volume('west1950', 1950),
        volume('reprint1950', 1950),
        volume('east1920', 1920),
        volume('paper', 1910, { digitized: false }),
      ],
      town.volumes,
    );
    expect(options.map((o) => [o.label, o.status])).toEqual([
      ['1950 (1)', 'placed'],
      ['1950 (2)', 'unplaced'],
      ['1920', 'unplaced'],
      ['1910', 'not digitized'],
    ]);
  });
});

describe('footprintFeatures', () => {
  it("draws each volume's display shape, and the selected one whole", () => {
    const towns = new Map([['s/town', town]]);
    const plain = footprintFeatures(towns, null);
    // The whole-town series is covered by the two volumes, so is not drawn.
    expect(plain.features.map((f) => f.id)).toEqual(['s/town#0', 's/town#1']);
    const selected = footprintFeatures(towns, { place: 's/town', index: 2 });
    const whole = selected.features.find((f) => f.id === 's/town#2');
    expect(whole?.properties?.selected).toBe(1);
    expect(whole?.geometry).toEqual({
      type: 'MultiPolygon',
      coordinates: square(0, 0, 10, 10),
    });
  });
});
