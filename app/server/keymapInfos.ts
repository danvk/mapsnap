/**
 * A volume's key-map sheets and the sidecars the viewer can use for each.
 *
 * Key maps are `<stem>.keymap.json`. Siblings: `<stem>.regions.panels.json`
 * (region view), `<stem>.georef.json` (the key map's own georeference, whose
 * `corners` place the sheet on the map) and `<stem>.roadprob.png` (the key
 * map's P(road) map, #211). The corners are what the key-map underlay draws
 * the sheet, or its P(road) map, with.
 *
 * A volume fitted locally keeps all of these in `raw/` beside the sheet. One
 * synced from the mirror (scripts/pull_item.py) does not: the sheet is in the
 * volume's `raw/`, but each run's key-map sidecars are in `runs/<tag>/raw/`
 * (#554). See keymapLocation.
 */
import { readdir, readFile, stat } from 'fs/promises';
import { join } from 'path';
import type { KeymapInfo } from './api.ts';

/** The four (lon, lat) corners of a georef document, or undefined if malformed. */
export function georefCorners(doc: unknown): [number, number][] | undefined {
  const corners = (doc as { corners?: unknown })?.corners;
  if (!Array.isArray(corners) || corners.length !== 4) return undefined;
  const points: [number, number][] = [];
  for (const corner of corners) {
    if (
      !Array.isArray(corner) ||
      corner.length < 2 ||
      typeof corner[0] !== 'number' ||
      typeof corner[1] !== 'number' ||
      !Number.isFinite(corner[0]) ||
      !Number.isFinite(corner[1])
    ) {
      return undefined;
    }
    points.push([corner[0], corner[1]]);
  }
  return points;
}

/** Files in a directory, or none when it does not exist. */
async function listDir(dir: string): Promise<string[]> {
  try {
    return await readdir(dir);
  } catch {
    return [];
  }
}

/**
 * Where a volume's key-map sidecars are, volume-relative: `raw` or `runs/<tag>/raw`.
 *
 * The selected `run`'s own directory when it holds a `*.keymap.json`, else
 * the volume's `raw/`, else the run that wrote one most recently: the key map
 * belongs to the volume, so a run that wrote none of its own (a republication,
 * say) still shows the one another run detected, and a fresh corpus run wins
 * over an old pilot. Null when no directory has one: the volume has no key maps.
 */
export async function keymapSidecarDir(
  volumeDir: string,
  run?: string,
): Promise<string | null> {
  for (const dir of [...(run ? [`${run}/raw`] : []), 'raw']) {
    if (await hasKeymap(join(volumeDir, dir))) return dir;
  }
  let newest: { dir: string; mtimeMs: number } | null = null;
  for (const name of await listDir(join(volumeDir, 'runs'))) {
    const dir = `runs/${name}/raw`;
    if (dir === `${run}/raw` || !(await hasKeymap(join(volumeDir, dir)))) {
      continue;
    }
    // The key maps' own times, not the directory's: a sync from the mirror
    // keeps each object's time but stamps every directory with the sync's.
    const mtimeMs = await newestKeymapTime(join(volumeDir, dir));
    if (!newest || mtimeMs > newest.mtimeMs) newest = { dir, mtimeMs };
  }
  return newest?.dir ?? null;
}

/** When the newest `*.keymap.json` in a directory was written, in ms. */
async function newestKeymapTime(dir: string): Promise<number> {
  let newest = 0;
  for (const file of await listDir(dir)) {
    if (!file.endsWith('.keymap.json')) continue;
    newest = Math.max(newest, (await stat(join(dir, file))).mtimeMs);
  }
  return newest;
}

/** Whether a directory holds any `*.keymap.json`. */
async function hasKeymap(dir: string): Promise<boolean> {
  return (await listDir(dir)).some((file) => file.endsWith('.keymap.json'));
}

/**
 * The volume-relative directory holding `file`: the sidecar directory, else the volume's `raw/`.
 *
 * A mirror run keeps sidecars under `runs/<tag>/raw/` but the sheet itself in
 * the volume's `raw/`, beside the other page scans.
 */
export async function keymapFileDir(
  volumeDir: string,
  sidecarDir: string,
  file: string,
): Promise<string | null> {
  for (const dir of [sidecarDir, 'raw']) {
    if ((await listDir(join(volumeDir, dir))).includes(file)) return dir;
  }
  return null;
}

/**
 * Key-map sheets of the volume at `volumeDir`, sorted by stem, with their sidecars.
 *
 * `serviceRoot` is the IIIF image service root for the volume
 * (`.../iiif/<volume>`); each key map's sheet and P(road) map are addressed
 * under it as absolute URLs, the way the annotation rewrite addresses page
 * tiles, so the viewer reaches the API server directly in development too
 * (the Vite proxy does not forward `/iiif`).
 *
 * `run` (`runs/<tag>`) is the run on screen, whose own key-map sidecars are
 * preferred (see keymapSidecarDir). Empty when the volume has no key maps.
 */
export async function keymapInfos(
  volumeDir: string,
  serviceRoot: string,
  run?: string,
): Promise<KeymapInfo[]> {
  const sidecarDir = await keymapSidecarDir(volumeDir, run);
  if (sidecarDir === null) return [];
  const files = await listDir(join(volumeDir, sidecarDir));
  const present = new Set(files);
  const infos: KeymapInfo[] = [];
  for (const file of files.filter((f) => f.endsWith('.keymap.json')).sort()) {
    const stem = file.slice(0, -'.keymap.json'.length);
    const hasGeoref = present.has(`${stem}.georef.json`);
    let corners: [number, number][] | undefined;
    if (hasGeoref) {
      try {
        corners = georefCorners(
          JSON.parse(
            await readFile(
              join(volumeDir, sidecarDir, `${stem}.georef.json`),
              'utf8',
            ),
          ),
        );
      } catch {
        corners = undefined;
      }
    }
    let image: { dir: string; file: string } | undefined;
    for (const name of [`${stem}.jpg`, `${stem}.png`]) {
      const dir = await keymapFileDir(volumeDir, sidecarDir, name);
      if (dir !== null) {
        image = { dir, file: name };
        break;
      }
    }
    const hasRoadprob = present.has(`${stem}.roadprob.png`);
    infos.push({
      stem,
      sidecarDir,
      ...(image ? { image: `${image.dir}/${image.file}` } : {}),
      hasRegions: present.has(`${stem}.regions.panels.json`),
      hasGeoref,
      hasRoadprob,
      ...(corners ? { corners } : {}),
      ...(image
        ? { imageService: `${serviceRoot}/${image.dir}/${image.file}` }
        : {}),
      ...(hasRoadprob
        ? {
            roadprobService: `${serviceRoot}/${sidecarDir}/${stem}.roadprob.png`,
          }
        : {}),
    });
  }
  return infos;
}
