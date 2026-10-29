import { describe, expect, it } from 'vitest';

import {
  chronoscopeImageUrl,
  itemViewerUrl,
  scaleForWidth,
  sheetOfStem,
  unplacedImages,
  type ItemRecord,
  type ItemSheet,
} from './itemRecord.ts';

const IMAGE =
  'https://cdn.chronoscope.io/mapsnap/service:gmd:x:00001_1907-0002';

function sheet(
  seq: number,
  key: string,
  extra: Partial<ItemSheet> = {},
): ItemSheet {
  return {
    seq,
    key,
    width: 1613,
    height: 1913,
    image: IMAGE,
    thumbnail: {
      url: `${IMAGE}/full/404,479/0/default.jpg`,
      width: 404,
      height: 479,
    },
    ...extra,
  };
}

// A rectangular panel outline.
function square(x0: number, y0: number, x1: number, y1: number) {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ] as [number, number][];
}

// Abbeville 1907: p1 placed whole; p2 split in three, one panel placed.
const record: ItemRecord = {
  id: 'sanborn00001_001',
  name: 'Abbeville, Henry County, Alabama',
  date: '1907-06',
  sheets: [
    sheet(1, 'p1', { fit: [{ image: 'p1', status: 'placed' }] }),
    sheet(2, 'p2', {
      fit: [
        { image: 'p2__1', status: 'placed' },
        { image: 'p2__2', status: 'unplaced' },
        { image: 'p2__3', status: 'withheld' },
      ],
      panels: [
        { points: square(0, 0, 800, 900) },
        {
          points: [
            [800, 0.4],
            [1613, 0],
            [1613, 900],
            [800, 900],
          ],
        },
        { points: square(0, 900, 1613, 1913) },
      ],
    }),
    sheet(3, 'p3', { fit: [{ image: 'p3', status: 'unplaced' }] }),
    sheet(4, 'p0a', { fit: [{ image: 'p0a', status: 'key map' }] }),
  ],
};

describe('unplacedImages', () => {
  const images = unplacedImages(record);

  it('lists unplaced and withheld images, but not key maps', () => {
    expect(images.map((image) => [image.image, image.status])).toEqual([
      ['p2__2', 'unplaced'],
      ['p2__3', 'withheld'],
      ['p3', 'unplaced'],
    ]);
  });

  it('cuts a panel out along its outline, and leaves a whole sheet whole', () => {
    expect(images[0]!.crop).toMatchObject({
      x: 800,
      y: 0,
      width: 813,
      height: 900,
    });
    expect(images[2]!.crop).toBeNull();
  });
});

describe('sheetOfStem', () => {
  it('finds the sheet a panel is on', () => {
    expect(sheetOfStem(record, 'p2__1')?.seq).toBe(2);
    expect(sheetOfStem(record, 'p9')).toBeNull();
  });
});

describe('itemViewerUrl', () => {
  it('links the item page, optionally at a sheet', () => {
    expect(itemViewerUrl('sanborn00001_001')).toBe(
      'https://mapsnap.org/item?id=sanborn00001_001',
    );
    expect(itemViewerUrl('sanborn00001_001', 2)).toBe(
      'https://mapsnap.org/item?id=sanborn00001_001&sp=2',
    );
  });
});

describe('chronoscopeImageUrl', () => {
  it("asks for one of the sizes the CDN's info.json lists", () => {
    const p1 = sheet(1, 'p1');
    expect(chronoscopeImageUrl(p1, 4)).toBe(
      `${IMAGE}/full/404,479/0/default.jpg`,
    );
    expect(chronoscopeImageUrl(p1, 2)).toBe(
      `${IMAGE}/full/807,957/0/default.jpg`,
    );
    // Its explicit size is a 404 on the CDN.
    expect(chronoscopeImageUrl(p1, 1)).toBe(`${IMAGE}/full/max/0/default.jpg`);
  });
});

describe('scaleForWidth', () => {
  it('picks the smallest size at least as wide as asked', () => {
    expect(scaleForWidth(1613, 300)).toBe(4);
    expect(scaleForWidth(1613, 404)).toBe(4);
    expect(scaleForWidth(1613, 405)).toBe(2);
    expect(scaleForWidth(1613, 1200)).toBe(1);
  });
});
