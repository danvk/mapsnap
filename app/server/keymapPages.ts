/**
 * Locating key-map pages and their truth sidecars under the data directory.
 *
 * A key map is a page in some volume's `raw/` directory carrying a
 * `<stem>.keymap.json` sidecar; its truth labels live beside it in
 * `raw/truth/<stem>.labels.json`. Volumes may be nested one level (e.g.
 * `data/queens_1950/vol2/raw/`), so a page is identified by the slash-joined
 * volume path and stem — "queens_1950/vol2/p0", "champaign_ill_1915/p1" —
 * which is also what the UI displays.
 */

import { readFile, readdir } from 'fs/promises';
import { join } from 'path';

/** Extensions a raw page image may use, in the order they are probed. */
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png'];

/** How many directory levels below `data/` a volume may sit (`data/queens_1950/vol2`). */
const MAX_VOLUME_DEPTH = 2;

/** A key-map page: where its files live and which of them exist. */
export interface KeymapPage {
  /** Slash-joined volume path and stem, e.g. "queens_1950/vol2/p0". */
  id: string;
  /** Volume directory relative to the data dir, e.g. "queens_1950/vol2". */
  volume: string;
  /** Page stem within `raw/`, e.g. "p0". */
  stem: string;
  /** Absolute path of the page image. */
  imagePath: string;
  /** Absolute path of the truth sidecar, which need not exist yet. */
  labelsPath: string;
  /** Whether the page has a `<stem>.keymap.json` detection sidecar. */
  hasKeymap: boolean;
}

// The page image for a stem, probing the known extensions; null if none exists.
function findImage(
  rawFiles: Set<string>,
  rawDir: string,
  stem: string,
): string | null {
  for (const extension of IMAGE_EXTENSIONS) {
    if (rawFiles.has(stem + extension)) return join(rawDir, stem + extension);
  }
  return null;
}

// Describe one page of a volume, or null when it has no image to label.
function pageFor(
  volume: string,
  rawDir: string,
  rawFiles: Set<string>,
  stem: string,
): KeymapPage | null {
  const imagePath = findImage(rawFiles, rawDir, stem);
  if (!imagePath) return null;
  return {
    id: `${volume}/${stem}`,
    volume,
    stem,
    imagePath,
    labelsPath: join(rawDir, 'truth', `${stem}.labels.json`),
    hasKeymap: rawFiles.has(`${stem}.keymap.json`),
  };
}

/**
 * Volume directories (relative to `dataDir`) that contain a `raw/` subdirectory.
 *
 * Descends at most {@link MAX_VOLUME_DEPTH} levels and stops at the first
 * `raw/`-bearing directory on each branch, so a volume's own subdirectories
 * (`oim/`, `artifacts/`, …) are never scanned.
 */
async function findVolumes(
  dataDir: string,
  relative = '',
  depth = 0,
): Promise<string[]> {
  let entries;
  try {
    entries = await readdir(join(dataDir, relative), { withFileTypes: true });
  } catch {
    return [];
  }
  const directories = entries.filter(
    (entry) => entry.isDirectory() && !entry.name.startsWith('.'),
  );
  if (relative && directories.some((entry) => entry.name === 'raw')) {
    return [relative];
  }
  if (depth >= MAX_VOLUME_DEPTH) return [];
  const volumes: string[] = [];
  for (const entry of directories) {
    if (entry.name === 'raw') continue;
    const path = relative ? `${relative}/${entry.name}` : entry.name;
    volumes.push(...(await findVolumes(dataDir, path, depth + 1)));
  }
  return volumes;
}

/**
 * Every key-map page under the data directory, sorted by id.
 *
 * A page qualifies if it has a `keymap.json` sidecar — the detection output
 * that marks a sheet as a key map — *or* an existing truth sidecar, so labels
 * written for a page whose key-map detection has not been run stay reachable
 * in the labeler rather than dropping out of the list.
 */
export async function findKeymapPages(dataDir: string): Promise<KeymapPage[]> {
  const pages: KeymapPage[] = [];
  for (const volume of await findVolumes(dataDir)) {
    const rawDir = join(dataDir, volume, 'raw');
    let rawFiles: string[];
    try {
      rawFiles = await readdir(rawDir);
    } catch {
      continue;
    }
    let truthFiles: string[] = [];
    try {
      truthFiles = await readdir(join(rawDir, 'truth'));
    } catch {
      // No truth directory yet; the volume's labeled pages are simply none.
    }
    const present = new Set(rawFiles);
    const stems = new Set([
      ...rawFiles
        .filter((file) => file.endsWith('.keymap.json'))
        .map((file) => file.slice(0, -'.keymap.json'.length)),
      ...truthFiles
        .filter((file) => file.endsWith('.labels.json'))
        .map((file) => file.slice(0, -'.labels.json'.length)),
    ]);
    for (const stem of stems) {
      const page = pageFor(volume, rawDir, present, stem);
      if (page) pages.push(page); // a sidecar with no page image is unlabelable
    }
  }
  pages.sort((a, b) => a.id.localeCompare(b.id));
  return pages;
}

/**
 * Resolve a page id to its file paths, or null if it does not name a page.
 *
 * Rejects ids that could escape the data directory (empty, `.`/`..` or
 * backslash-bearing segments) and ids with no page image, so a non-null result
 * is always a real page under `dataDir`.
 */
export async function resolveKeymapPage(
  dataDir: string,
  id: string,
): Promise<KeymapPage | null> {
  const parts = id.split('/');
  if (parts.length < 2 || parts.length > MAX_VOLUME_DEPTH + 1) return null;
  const unsafe = (part: string) =>
    part === '' || part === '.' || part === '..' || part.includes('\\');
  if (parts.some(unsafe)) return null;
  const stem = parts[parts.length - 1]!;
  const volume = parts.slice(0, -1).join('/');
  const rawDir = join(dataDir, volume, 'raw');
  let rawFiles: string[];
  try {
    rawFiles = await readdir(rawDir);
  } catch {
    return null;
  }
  return pageFor(volume, rawDir, new Set(rawFiles), stem);
}

/** Label counts in a page's truth sidecar, split by whether text was entered.

Zeros when the sidecar is missing or unreadable — visually equivalent, since
a zero count draws no badge. */
export async function readLabelCounts(
  labelsPath: string,
): Promise<{ withText: number; withoutText: number }> {
  try {
    const data = JSON.parse(await readFile(labelsPath, 'utf8'));
    const labels: { text?: unknown }[] = Array.isArray(data.labels)
      ? data.labels
      : [];
    const withText = labels.filter((l) => String(l.text ?? '').trim()).length;
    return { withText, withoutText: labels.length - withText };
  } catch {
    return { withText: 0, withoutText: 0 };
  }
}
