import { mkdir, mkdtemp, readFile, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  findVolumes,
  isSafePage,
  isSafeVolume,
  pageSortKey,
  readTruth,
  truthPath,
  volumePages,
  writePageTruth,
} from './adjacencyTruth.ts';

let dataDir: string;

beforeAll(async () => {
  dataDir = await mkdtemp(join(tmpdir(), 'adjacency-truth-'));
  const write = async (relative: string, contents = 'jpeg') => {
    const path = join(dataDir, relative);
    await mkdir(join(path, '..'), { recursive: true });
    await writeFile(path, contents);
  };
  // A plain volume with whole sheets, a split panel, and non-page files.
  for (const name of ['p1.jpg', 'p2.jpg', 'p10.jpg', 'p33B.jpg', 'p6__1.jpg']) {
    await write(`champaign/${name}`);
  }
  await write('champaign/raw/p0.jpg'); // raw/ must not be listed as a volume
  await write('champaign/manifest.json', '{}');
  // A nested volume.
  await write('queens_1950/vol2/p0.jpg');
  await write('queens_1950/vol2/p5.jpg');
  // A directory with no pages at all.
  await write('loc/readme.txt', 'x');
});

describe('discovery', () => {
  it('finds plain and nested volumes, not their subdirectories', async () => {
    expect(await findVolumes(dataDir)).toEqual([
      'champaign',
      'queens_1950/vol2',
    ]);
  });

  it('lists whole-sheet stems in natural order, excluding split panels', async () => {
    expect(await volumePages(dataDir, 'champaign')).toEqual([
      'p1',
      'p2',
      'p10',
      'p33B',
    ]);
  });
});

describe('validation', () => {
  it('accepts real names and rejects traversal', () => {
    expect(isSafeVolume('champaign')).toBe(true);
    expect(isSafeVolume('queens_1950/vol2')).toBe(true);
    for (const bad of ['', '..', 'a/../b', 'a/b/c', 'a\\b', '/abs']) {
      expect(isSafeVolume(bad)).toBe(false);
    }
    expect(isSafePage('p12')).toBe(true);
    expect(isSafePage('p33B')).toBe(true);
    for (const bad of ['p6__1', '12', 'p12/x', '..', 'adjacency-truth']) {
      expect(isSafePage(bad)).toBe(false);
    }
  });

  it('natural sort puts p2 before p10', () => {
    const stems = ['p10', 'p2', 'p33B', 'p33A'];
    stems.sort((a, b) => {
      const [na, sa] = pageSortKey(a);
      const [nb, sb] = pageSortKey(b);
      return na - nb || sa.localeCompare(sb);
    });
    expect(stems).toEqual(['p2', 'p10', 'p33A', 'p33B']);
  });
});

describe('truth round-trip', () => {
  it('reads empty for a volume with no truth file', async () => {
    expect(await readTruth(dataDir, 'champaign')).toEqual({});
  });

  it('writes one page without disturbing others, in natural order', async () => {
    const entry = (text: string) => ({
      width: 1665,
      height: 2018,
      labels: [{ x: 10, y: 20, text }],
    });
    await writePageTruth(dataDir, 'champaign', 'p10', entry('11'));
    await writePageTruth(dataDir, 'champaign', 'p2', entry('3'));
    const pages = await readTruth(dataDir, 'champaign');
    expect(Object.keys(pages)).toEqual(['p2', 'p10']); // natural order on disk
    expect(pages['p10']!.labels[0]!.text).toBe('11');
    // Overwriting one page keeps the other.
    await writePageTruth(dataDir, 'champaign', 'p2', entry('4'));
    const after = await readTruth(dataDir, 'champaign');
    expect(after['p2']!.labels[0]!.text).toBe('4');
    expect(after['p10']!.labels[0]!.text).toBe('11');
    // The file itself lives beside the volume's pages.
    const raw = JSON.parse(
      await readFile(truthPath(dataDir, 'champaign'), 'utf8'),
    );
    expect(Object.keys(raw.pages)).toEqual(['p2', 'p10']);
  });
});
