import { mkdir, mkdtemp, readFile, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  comparePages,
  findVolumes,
  isSafePage,
  isSafeVolume,
  isSupersededSheet,
  panelParent,
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
  // A plain volume with whole sheets, a split sheet (p6 + its panels), an
  // orphan panel whose sheet is not on disk, and non-page files.
  for (const name of [
    'p1.jpg',
    'p2.jpg',
    'p10.jpg',
    'p33B.jpg',
    'p6.jpg',
    'p6__1.jpg',
    'p6__2.jpg',
    'p7__1.jpg',
  ]) {
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

  it('keeps a superseded sheet listed when it still carries labels', async () => {
    // p6 is split, but its own labels must stay reachable in the UI.
    expect(await volumePages(dataDir, 'champaign', new Set(['p6']))).toEqual([
      'p1',
      'p2',
      'p6',
      'p6__1',
      'p6__2',
      'p7__1',
      'p10',
      'p33B',
    ]);
  });

  it('flags a sheet whose panels supersede it', async () => {
    const pages = await volumePages(dataDir, 'champaign', new Set(['p6']));
    expect(isSupersededSheet('p6', pages)).toBe(true);
    expect(isSupersededSheet('p1', pages)).toBe(false);
    expect(isSupersededSheet('p6__1', pages)).toBe(false);
  });

  it('lists panels instead of the sheets they split, in natural order', async () => {
    expect(await volumePages(dataDir, 'champaign')).toEqual([
      'p1',
      'p2',
      'p6__1', // p6 itself is not offered: its panels supersede it
      'p6__2',
      'p7__1', // a panel whose sheet is absent still stands on its own
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
    expect(isSafePage('p12__1')).toBe(true);
    for (const bad of [
      '12',
      'p12/x',
      '..',
      'adjacency-truth',
      'p12__',
      'p12__x',
    ]) {
      expect(isSafePage(bad)).toBe(false);
    }
  });

  it('natural sort orders numbers, suffixes, then panels', () => {
    const stems = ['p10', 'p2', 'p33B', 'p33A', 'p10__2', 'p10__1'];
    stems.sort(comparePages);
    expect(stems).toEqual(['p2', 'p10', 'p10__1', 'p10__2', 'p33A', 'p33B']);
  });

  it('panelParent strips a panel index and leaves sheets alone', () => {
    expect(panelParent('p1401__2')).toBe('p1401');
    expect(panelParent('p1401')).toBe('p1401');
    expect(panelParent('p33B')).toBe('p33B');
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
