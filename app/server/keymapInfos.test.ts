import { mkdir, mkdtemp, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeAll, describe, expect, it } from 'vitest';
import { georefCorners, keymapInfos } from './keymapInfos.ts';

const corners = [
  [-90.1158, 29.9432],
  [-90.0872, 29.9745],
  [-90.0431, 29.9458],
  [-90.0717, 29.9145],
];

let rawDir: string;
const SERVICE = 'http://localhost:8182/iiif/new_orleans/raw';

// A raw/ directory shaped like New Orleans 1896's: one key map with every
// sidecar, a second (Los Angeles-style pb) with a georef but no P(road) map,
// and a third with a georef whose corners are malformed.
beforeAll(async () => {
  rawDir = await mkdtemp(join(tmpdir(), 'keymap-infos-'));
  const write = (name: string, contents = '{}') =>
    writeFile(join(rawDir, name), contents);
  await write('p0.jpg', 'jpeg');
  await write('p0.keymap.json');
  await write('p0.regions.panels.json');
  await write('p0.georef.json', JSON.stringify({ corners, width: 6397 }));
  await write('p0.roadprob.png', 'png');
  await write('pb.png', 'png');
  await write('pb.keymap.json');
  await write('pb.georef.json', JSON.stringify({ corners }));
  await write('pz.keymap.json');
  await write(
    'pz.georef.json',
    JSON.stringify({ corners: corners.slice(0, 3) }),
  );
  await write('p7.georef.json', JSON.stringify({ corners })); // not a key map
  await mkdir(join(rawDir, 'truth'));
});

describe('keymapInfos', () => {
  it('lists key maps with their sidecars and georef corners', async () => {
    const infos = await keymapInfos(rawDir, SERVICE);
    expect(infos.map((info) => info.stem)).toEqual(['p0', 'pb', 'pz']);
    expect(infos[0]).toEqual({
      stem: 'p0',
      hasRegions: true,
      hasGeoref: true,
      hasRoadprob: true,
      corners,
      imageService: `${SERVICE}/p0.jpg`,
      roadprobService: `${SERVICE}/p0.roadprob.png`,
    });
    // A PNG key map (Queens) is addressed as one; no P(road) map, no service.
    expect(infos[1]).toEqual({
      stem: 'pb',
      hasRegions: false,
      hasGeoref: true,
      hasRoadprob: false,
      corners,
      imageService: `${SERVICE}/pb.png`,
    });
  });

  it('omits corners from a georef that lacks four points', async () => {
    const infos = await keymapInfos(rawDir, SERVICE);
    expect(infos[2].hasGeoref).toBe(true);
    expect(infos[2].corners).toBeUndefined();
  });

  it('is empty for a volume without a raw directory', async () => {
    expect(await keymapInfos(join(rawDir, 'missing'), SERVICE)).toEqual([]);
  });
});

describe('georefCorners', () => {
  it('accepts four finite lon/lat pairs and nothing else', () => {
    expect(georefCorners({ corners })).toEqual(corners);
    expect(
      georefCorners({
        corners: [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      }),
    ).toBeUndefined();
    expect(
      georefCorners({ corners: [[1, 2], [3, 4], [5, 6], [7]] }),
    ).toBeUndefined();
    expect(
      georefCorners({
        corners: [
          [1, 2],
          [3, 4],
          [5, 6],
          ['7', 8],
        ],
      }),
    ).toBeUndefined();
    expect(georefCorners({})).toBeUndefined();
    expect(georefCorners(null)).toBeUndefined();
  });
});
