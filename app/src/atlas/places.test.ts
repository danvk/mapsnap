import { describe, expect, it } from 'vitest';

import {
  annotationUri,
  defaultYear,
  locItemUrl,
  locSheetUrl,
  locStemOf,
  searchPlaces,
  stateSlug,
  volumesOfYear,
  yearsOf,
  type Place,
  type Volume,
} from './places.ts';

const index = { bucket: 'mapsnap-sanborn', runTag: 'corpus-v1' };

function volume(item: string, year: number | null, mirror?: string): Volume {
  return {
    item,
    date: year ? String(year) : '',
    year,
    sheets: 10,
    title: '',
    ...(mirror ? { state: 'illinois', mirrorYear: mirror } : {}),
  };
}

function place(name: string, state: string, sheets: number): Place {
  return {
    id: `${state.toLowerCase()}/${name.toLowerCase()}`,
    name,
    state,
    lon: 0,
    lat: 0,
    volumes: 1,
    sheets,
    mirrored: 1,
    firstYear: 1900,
    lastYear: 1950,
  };
}

describe('annotationUri', () => {
  it('builds the run key from the mirror prefix, not the catalogue date', () => {
    // sanborn00518_001 is catalogued 1890 and mirrored under 1899; 310 items
    // disagree this way, and a URI built from the date 404s.
    const stale = volume('sanborn00518_001', 1890, '1899');
    expect(annotationUri(stale, index)).toBe(
      's3://mapsnap-sanborn/by-state/illinois/1899/sanborn00518_001/' +
        'runs/corpus-v1/mapsnap.iiif.json',
    );
  });

  it('has no URI for a volume the mirror never took', () => {
    expect(annotationUri(volume('sanborn99999_001', 1912), index)).toBeNull();
  });
});

describe('yearsOf', () => {
  it('is newest first, with undated volumes last', () => {
    const volumes = [
      volume('a', 1912),
      volume('b', null),
      volume('c', 1950),
      volume('d', 1912),
    ];
    expect(yearsOf(volumes)).toEqual([1950, 1912, null]);
  });
});

describe('defaultYear', () => {
  it('opens on the newest year that has something to draw', () => {
    // 1960 is catalogued but never mirrored; opening there would render an
    // empty map, which reads as a broken app rather than a gap in the data.
    const volumes = [volume('new', 1960), volume('old', 1950, '1950')];
    expect(defaultYear(volumes)).toBe(1950);
  });

  it('falls back to the newest of any kind when nothing is mirrored', () => {
    expect(defaultYear([volume('a', 1901), volume('b', 1960)])).toBe(1960);
  });

  it('is null for a town with no volumes', () => {
    expect(defaultYear([])).toBeNull();
  });
});

describe('volumesOfYear', () => {
  it('takes every volume of a year, since a city-year is often several', () => {
    const volumes = [volume('a', 1950), volume('b', 1950), volume('c', 1912)];
    expect(volumesOfYear(volumes, 1950).map((v) => v.item)).toEqual(['a', 'b']);
    expect(volumesOfYear(volumes, null)).toEqual([]);
  });
});

describe('searchPlaces', () => {
  const places = [
    place('Chicopee', 'Massachusetts', 200),
    place('Chicago', 'Illinois', 12000),
    place('East Chicago', 'Indiana', 400),
    place('Peoria', 'Illinois', 900),
  ];

  it('ranks a prefix match above an interior one, whatever the sizes', () => {
    expect(searchPlaces(places, 'chic').map((p) => p.name)).toEqual([
      'Chicago',
      'Chicopee',
      'East Chicago',
    ]);
  });

  it('matches the state too, so "chicago, il" works', () => {
    expect(searchPlaces(places, 'chicago, il')[0]?.name).toBe('Chicago');
    expect(searchPlaces(places, 'illinois').map((p) => p.name)).toEqual([
      'Chicago',
      'Peoria',
    ]);
  });

  it('offers nothing for an empty query', () => {
    expect(searchPlaces(places, '   ')).toEqual([]);
  });

  it('caps the list', () => {
    expect(searchPlaces(places, 'o', 2)).toHaveLength(2);
  });
});

describe('stateSlug', () => {
  it('is the volumes file a place needs', () => {
    expect(stateSlug('illinois/chicago')).toBe('illinois');
    expect(stateSlug('new-york/new-york')).toBe('new-york');
  });
});

// South Bend 1917, whose p28 is the 33rd image of its LoC resource.
const southBend: Volume = {
  item: 'sanborn02502_005',
  date: '1917',
  year: 1917,
  sheets: 123,
  title: 'South Bend',
  loc: {
    resource: 'g4094sm.g4094sm_g025021917',
    prefix: '02502_1917-',
    sheets: [
      'titl',
      'ind1',
      'ind2',
      'cbd1',
      '0000',
      '0001',
      '0002',
      '0003',
      '0004',
      '0005',
      '0006',
      '0007',
      '0008',
      '0009',
      '0010',
      '0011',
      '0012',
      '0013',
      '0014',
      '0015',
      '0016',
      '0017',
      '0018',
      '0019',
      '0020',
      '0021',
      '0022',
      '0023',
      '0024',
      '0025',
      '0026',
      '0027',
      '0028',
      '0029',
    ],
  },
};
const P28 =
  'service:gmd:gmd409m:g4094m:g4094sm:g4094sm_g025021917:02502_1917-0028';

describe('locStemOf', () => {
  it('reads the stem off a loc.gov service or the CDN copy of it', () => {
    expect(
      locStemOf(`https://tile.loc.gov/image-services/iiif/${P28}/info.json`),
    ).toBe('02502_1917-0028');
    expect(locStemOf(`https://cdn.chronoscope.io/mapsnap/${P28}`)).toBe(
      '02502_1917-0028',
    );
  });

  it("is null for a service that is not LoC's", () => {
    expect(locStemOf('http://localhost:8182/iiif/vol/p28.jpg')).toBeNull();
    expect(locStemOf(undefined)).toBeNull();
  });
});

describe('locSheetUrl', () => {
  it("links to the sheet's own page of the LoC resource", () => {
    expect(
      locSheetUrl(southBend, `https://cdn.chronoscope.io/mapsnap/${P28}`),
    ).toBe('https://www.loc.gov/resource/g4094sm.g4094sm_g025021917/?sp=33');
  });

  it('falls back to the item page when the sheet cannot be found', () => {
    const itemPage = 'https://www.loc.gov/item/sanborn02502_005/';
    expect(locItemUrl('sanborn02502_005')).toBe(itemPage);
    expect(locSheetUrl({ ...southBend, loc: undefined }, P28)).toBe(itemPage);
    expect(locSheetUrl(southBend, P28.replace('0028', '0099'))).toBe(itemPage);
    expect(locSheetUrl(southBend, undefined)).toBe(itemPage);
  });
});
