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
 * A sheet that has been split into panels is labelled panel by panel, so its
 * keys are the panel stems (`p1401__1`, `p1401__2`) and the composite sheet is
 * not offered: the pipeline georeferences panels, and a printed neighbor number
 * along a panel's edge describes that panel's neighborhood, not the sheet's.
 *
 * Coordinates are in the pixel frame of the volume-root page image (the 25%
 * scan the debugger displays), recorded by each entry's width/height.
 */

import { readFile, readdir, writeFile } from 'fs/promises';
import { join } from 'path';

/** A page image at the volume root: a whole sheet (p12, p33B, p101w) or a panel (p12__1). */
const PAGE_IMAGE = /^p\d+[A-Za-z]{0,2}(?:__\d+)?\.jpe?g$/;

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

/** Reject a page stem that is not a whole-sheet stem or a split panel. */
export function isSafePage(stem: string): boolean {
  return /^p\d+[A-Za-z]{0,2}(?:__\d+)?$/.test(stem);
}

/** Natural sort key for a page stem: number, letter suffix, then panel index. */
export function pageSortKey(stem: string): [number, string, number] {
  const match = /^p(\d+)([A-Za-z]*)(?:__(\d+))?$/.exec(stem);
  if (!match) return [Number.MAX_SAFE_INTEGER, stem, 0];
  return [Number(match[1]), match[2]!, match[3] ? Number(match[3]) : 0];
}

/** Compare page stems in natural order: p2 < p10 < p10__1 < p10__2 < p33A. */
export function comparePages(a: string, b: string): number {
  const [numberA, suffixA, panelA] = pageSortKey(a);
  const [numberB, suffixB, panelB] = pageSortKey(b);
  return numberA - numberB || suffixA.localeCompare(suffixB) || panelA - panelB;
}

/** The sheet a stem belongs to: a panel's parent (`p12__1` -> `p12`), else itself. */
export function panelParent(stem: string): string {
  const separator = stem.indexOf('__');
  return separator === -1 ? stem : stem.slice(0, separator);
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
      .sort(comparePages)
      .map((key) => [key, pages[key]]),
  );
  await writeFile(
    truthPath(dataDir, volume),
    JSON.stringify({ pages: sorted }, null, 2) + '\n',
  );
}

/**
 * Labelable page stems of a volume, naturally sorted.
 *
 * A sheet that has been split contributes its panels *instead of* itself, since
 * the panels are what carry a page's printed neighbor numbers into the
 * pipeline; a sheet with no panels on disk contributes itself. `keep` names
 * stems to list even when their panels supersede them — the caller passes the
 * sheets that already carry truth labels, so hand-made labels never vanish
 * from the UI when a sheet is split (they sort just above their panels).
 */
export async function volumePages(
  dataDir: string,
  volume: string,
  keep: ReadonlySet<string> = new Set(),
): Promise<string[]> {
  let files: string[];
  try {
    files = await readdir(join(dataDir, volume));
  } catch {
    return [];
  }
  const stems = files
    .filter((file) => PAGE_IMAGE.test(file))
    .map((file) => file.replace(/\.jpe?g$/, ''));
  const split = new Set(
    stems.filter((stem) => stem.includes('__')).map(panelParent),
  );
  return stems
    .filter((stem) => stem.includes('__') || !split.has(stem) || keep.has(stem))
    .sort(comparePages);
}

/** Whether a stem names a sheet that the volume has split into panels. */
export function isSupersededSheet(stem: string, pages: string[]): boolean {
  return (
    !stem.includes('__') &&
    pages.some((p) => panelParent(p) === stem && p !== stem)
  );
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
