import { mkdir, mkdtemp, utimes, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  georefCorners,
  keymapFileDir,
  keymapInfos,
  keymapSidecarDir,
} from './keymapInfos.ts';

const corners = [
  [-90.1158, 29.9432],
  [-90.0872, 29.9745],
  [-90.0431, 29.9458],
  [-90.0717, 29.9145],
];

let volumeDir: string;
let rawDir: string;
let mirrorDir: string;
const ROOT = 'http://localhost:8182/iiif/new_orleans';
const SERVICE = `${ROOT}/raw`;

// A raw/ directory shaped like New Orleans 1896's: one key map with every
// sidecar, a second (Los Angeles-style pb) with a georef but no P(road) map,
// and a third with a georef whose corners are malformed.
beforeAll(async () => {
  volumeDir = await mkdtemp(join(tmpdir(), 'keymap-infos-'));
  rawDir = join(volumeDir, 'raw');
  await mkdir(rawDir);
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

  // A volume synced from the mirror (#554): the sheet in the volume's raw/,
  // the key map's sidecars in each run's raw/. An older pilot run that sorts
  // first by name, and a later run that wrote no key map of its own.
  mirrorDir = await mkdtemp(join(tmpdir(), 'keymap-infos-mirror-'));
  await mkdir(join(mirrorDir, 'raw'));
  await writeFile(join(mirrorDir, 'raw', 'p0.jpg'), 'jpeg');
  await writeFile(join(mirrorDir, 'raw', 'p0.boxes.json'), '{}');
  const run = join(mirrorDir, 'runs', 'corpus-v1', 'raw');
  await mkdir(run, { recursive: true });
  await writeFile(join(run, 'p0.keymap.json'), '{}');
  await writeFile(join(run, 'p0.regions.panels.json'), '{}');
  await writeFile(join(run, 'p0.georef.json'), JSON.stringify({ corners }));
  await mkdir(join(mirrorDir, 'runs', 'another-run'), { recursive: true });
  const pilot = join(mirrorDir, 'runs', 'batch-test', 'raw');
  await mkdir(pilot, { recursive: true });
  await writeFile(join(pilot, 'p0.keymap.json'), '{}');
  const lastWeek = new Date(Date.now() - 7 * 86_400_000);
  await utimes(join(pilot, 'p0.keymap.json'), lastWeek, lastWeek);
});

describe('keymapInfos', () => {
  it('lists key maps with their sidecars and georef corners', async () => {
    const infos = await keymapInfos(volumeDir, ROOT);
    expect(infos.map((info) => info.stem)).toEqual(['p0', 'pb', 'pz']);
    expect(infos[0]).toEqual({
      stem: 'p0',
      sidecarDir: 'raw',
      image: 'raw/p0.jpg',
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
      sidecarDir: 'raw',
      image: 'raw/pb.png',
      hasRegions: false,
      hasGeoref: true,
      hasRoadprob: false,
      corners,
      imageService: `${SERVICE}/pb.png`,
    });
  });

  it('omits corners from a georef that lacks four points', async () => {
    const infos = await keymapInfos(volumeDir, ROOT);
    expect(infos[2].hasGeoref).toBe(true);
    expect(infos[2].corners).toBeUndefined();
  });

  it('is empty for a volume without a raw directory', async () => {
    expect(await keymapInfos(join(volumeDir, 'missing'), ROOT)).toEqual([]);
  });

  it("finds a mirror run's sidecars beside the volume's sheet (#554)", async () => {
    const infos = await keymapInfos(mirrorDir, ROOT, 'runs/corpus-v1');
    expect(infos).toEqual([
      {
        stem: 'p0',
        sidecarDir: 'runs/corpus-v1/raw',
        image: 'raw/p0.jpg',
        hasRegions: true,
        hasGeoref: true,
        hasRoadprob: false,
        corners,
        imageService: `${ROOT}/raw/p0.jpg`,
      },
    ]);
  });

  it("falls back to the newest run's key map when the run on screen has none", async () => {
    expect(await keymapSidecarDir(mirrorDir, 'runs/another-run')).toBe(
      'runs/corpus-v1/raw',
    );
    expect(await keymapSidecarDir(mirrorDir)).toBe('runs/corpus-v1/raw');
    expect(await keymapSidecarDir(volumeDir, 'runs/missing')).toBe('raw');
    expect(await keymapSidecarDir(join(volumeDir, 'missing'))).toBeNull();
  });

  it('looks for a file beside the sidecars, then in the raw directory', async () => {
    expect(await keymapFileDir(mirrorDir, 'runs/corpus-v1/raw', 'p0.jpg')).toBe(
      'raw',
    );
    expect(
      await keymapFileDir(mirrorDir, 'runs/corpus-v1/raw', 'p0.georef.json'),
    ).toBe('runs/corpus-v1/raw');
    expect(
      await keymapFileDir(mirrorDir, 'runs/corpus-v1/raw', 'p9.jpg'),
    ).toBeNull();
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
