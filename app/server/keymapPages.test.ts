import { mkdir, mkdtemp, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  findKeymapPages,
  readLabelCount,
  resolveKeymapPage,
} from './keymapPages.ts';

let dataDir: string;

// A data directory covering the layouts the labeler has to handle: a plain
// volume, a nested one, a page listed only for its truth labels, a keymap
// sidecar with no image, and a non-key-map page.
beforeAll(async () => {
  dataDir = await mkdtemp(join(tmpdir(), 'keymap-pages-'));
  const write = async (relative: string, contents = '{}') => {
    const path = join(dataDir, relative);
    await mkdir(join(path, '..'), { recursive: true });
    await writeFile(path, contents);
  };
  await write('champaign/raw/p1.jpg', 'jpeg');
  await write('champaign/raw/p1.keymap.json');
  await write('champaign/raw/truth/p1.labels.json', '{"labels":[{},{},{}]}');
  await write('champaign/raw/p7.jpg', 'jpeg'); // an ordinary page, not a key map
  await write('queens_1950/vol2/raw/p0.png', 'png');
  await write('queens_1950/vol2/raw/p0.keymap.json');
  await write('detroit/raw/p0.jpg', 'jpeg'); // truth only: no keymap.json
  await write('detroit/raw/truth/p0.labels.json', '{"labels":[]}');
  await write('detroit/raw/p9.keymap.json'); // no image alongside it
  await write('detroit/oim/p0.keymap.json'); // outside raw/: not a key map
});

describe('findKeymapPages', () => {
  it('finds key maps in plain and nested volumes, ignoring other pages', async () => {
    const pages = await findKeymapPages(dataDir);
    expect(pages.map((page) => page.id)).toEqual([
      'champaign/p1',
      'detroit/p0',
      'queens_1950/vol2/p0',
    ]);
  });

  it('reports whether each page has a keymap.json', async () => {
    const pages = await findKeymapPages(dataDir);
    const byId = Object.fromEntries(pages.map((p) => [p.id, p.hasKeymap]));
    expect(byId).toEqual({
      'champaign/p1': true,
      'queens_1950/vol2/p0': true,
      // Listed only because it has truth labels.
      'detroit/p0': false,
    });
  });

  it('points at the truth sidecar and the image, whatever its extension', async () => {
    const pages = await findKeymapPages(dataDir);
    const nested = pages.find((page) => page.id === 'queens_1950/vol2/p0')!;
    expect(nested.imagePath).toBe(join(dataDir, 'queens_1950/vol2/raw/p0.png'));
    expect(nested.labelsPath).toBe(
      join(dataDir, 'queens_1950/vol2/raw/truth/p0.labels.json'),
    );
  });
});

describe('resolveKeymapPage', () => {
  it('resolves plain and nested page ids', async () => {
    expect((await resolveKeymapPage(dataDir, 'champaign/p1'))?.stem).toBe('p1');
    const nested = await resolveKeymapPage(dataDir, 'queens_1950/vol2/p0');
    expect(nested?.volume).toBe('queens_1950/vol2');
  });

  it('resolves a page that is not a key map, since only ids are checked', async () => {
    const page = await resolveKeymapPage(dataDir, 'champaign/p7');
    expect(page?.hasKeymap).toBe(false);
  });

  it('rejects ids that name no image', async () => {
    expect(await resolveKeymapPage(dataDir, 'detroit/p9')).toBeNull();
    expect(await resolveKeymapPage(dataDir, 'champaign/nope')).toBeNull();
    expect(await resolveKeymapPage(dataDir, 'novolume/p1')).toBeNull();
  });

  it('rejects ids that could escape the data directory', async () => {
    for (const id of [
      '../champaign/p1',
      'champaign/../../p1',
      'champaign/raw/../p1',
      '/etc/passwd',
      'champaign',
      '',
      'a/b/c/d',
      'champaign\\p1',
    ]) {
      expect(await resolveKeymapPage(dataDir, id)).toBeNull();
    }
  });
});

describe('readLabelCount', () => {
  it('counts labels, and reports null when there is no sidecar', async () => {
    expect(
      await readLabelCount(join(dataDir, 'champaign/raw/truth/p1.labels.json')),
    ).toBe(3);
    expect(
      await readLabelCount(join(dataDir, 'detroit/raw/truth/p0.labels.json')),
    ).toBe(0);
    expect(await readLabelCount(join(dataDir, 'nope.labels.json'))).toBeNull();
  });
});
