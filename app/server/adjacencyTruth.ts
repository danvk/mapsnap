/**
 * Storage and discovery for hand-labelled adjacency truth.
 *
 * Sanborn sheets print the numbers of neighboring pages along their edges; the
 * adjacency labeler records those printed labels as (x, y, text) points, page
 * by page. Truth for a whole volume lives in one
 * ``<volume>/adjacency-truth.json`` — a sibling of the pipeline's
 * ``adjacency.json`` — keyed by page stem:
 *
 *   { "pages": { "p12": { "width": 1665, "height": 2018,
 *                          "labels": [{"x":…, "y":…, "text": "13"}] } } }
 *
 * Coordinates are in the pixel frame of the volume-root page image (the 25%
 * scan the debugger displays), recorded by each entry's width/height.
 */

import { readFile, readdir, writeFile } from 'fs/promises';
import { join } from 'path';

/** A whole-sheet page image at the volume root: p12, p33B, p101w — not splits. */
const PAGE_IMAGE = /^p\d+[A-Za-z]{0,2}\.jpe?g$/;

/** How many directory levels below `data/` a volume may sit (`data/queens_1950/vol2`). */
const MAX_VOLUME_DEPTH = 2;

/** One volume that has labelable pages. */
export interface AdjacencyVolume {
  /** Volume directory relative to the data dir, e.g. "queens_1950/vol2". */
  name: string;
  pageCount: number;
  /** Pages with at least one adjacency-truth label. */
  labeledPages: number;
}

/** A page entry in a volume's adjacency-truth.json. */
export interface AdjacencyPageTruth {
  width: number;
  height: number;
  labels: { x: number; y: number; text: string }[];
}

/** Reject a volume path that could escape the data directory. */
export function isSafeVolume(volume: string): boolean {
  const parts = volume.split('/');
  if (parts.length < 1 || parts.length > MAX_VOLUME_DEPTH) return false;
  return parts.every(
    (part) =>
      part !== '' && part !== '.' && part !== '..' && !part.includes('\\'),
  );
}

/** Reject a page stem that is not a plain whole-sheet stem. */
export function isSafePage(stem: string): boolean {
  return /^p\d+[A-Za-z]{0,2}$/.test(stem);
}

/** Natural sort key for page stems, so p2 sorts before p10. */
export function pageSortKey(stem: string): [number, string] {
  const match = /^p(\d+)([A-Za-z]*)$/.exec(stem);
  return match ? [Number(match[1]), match[2]] : [Number.MAX_SAFE_INTEGER, stem];
}

/** Path of a volume's adjacency truth file. */
export function truthPath(dataDir: string, volume: string): string {
  return join(dataDir, volume, 'adjacency-truth.json');
}

/** Read a volume's truth file; an empty mapping when it does not exist. */
export async function readTruth(
  dataDir: string,
  volume: string,
): Promise<Record<string, AdjacencyPageTruth>> {
  try {
    const doc = JSON.parse(await readFile(truthPath(dataDir, volume), 'utf8'));
    return doc.pages ?? {};
  } catch {
    return {};
  }
}

/** Write one page's labels into the volume's truth file (read-modify-write). */
export async function writePageTruth(
  dataDir: string,
  volume: string,
  page: string,
  entry: AdjacencyPageTruth,
): Promise<void> {
  const pages = await readTruth(dataDir, volume);
  pages[page] = entry;
  // Stable natural page order keeps the file diff-friendly across sessions.
  const sorted = Object.fromEntries(
    Object.keys(pages)
      .sort((a, b) => {
        const [na, sa] = pageSortKey(a);
        const [nb, sb] = pageSortKey(b);
        return na - nb || sa.localeCompare(sb);
      })
      .map((key) => [key, pages[key]]),
  );
  await writeFile(
    truthPath(dataDir, volume),
    JSON.stringify({ pages: sorted }, null, 2) + '\n',
  );
}

/** Whole-sheet page stems of a volume, naturally sorted. */
export async function volumePages(
  dataDir: string,
  volume: string,
): Promise<string[]> {
  let files: string[];
  try {
    files = await readdir(join(dataDir, volume));
  } catch {
    return [];
  }
  return files
    .filter((file) => PAGE_IMAGE.test(file))
    .map((file) => file.replace(/\.jpe?g$/, ''))
    .sort((a, b) => {
      const [na, sa] = pageSortKey(a);
      const [nb, sb] = pageSortKey(b);
      return na - nb || sa.localeCompare(sb);
    });
}

/**
 * Volumes (relative to `dataDir`) that have whole-sheet page images.
 *
 * Descends at most {@link MAX_VOLUME_DEPTH} levels and stops at the first
 * page-bearing directory on each branch, so a volume's own subdirectories
 * (`raw/`, `oim/`, …) are never listed as volumes themselves.
 */
export async function findVolumes(
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
  if (relative && entries.some((e) => e.isFile() && PAGE_IMAGE.test(e.name))) {
    return [relative];
  }
  if (depth >= MAX_VOLUME_DEPTH) return [];
  const volumes: string[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const path = relative ? `${relative}/${entry.name}` : entry.name;
    volumes.push(...(await findVolumes(dataDir, path, depth + 1)));
  }
  return volumes.sort();
}
