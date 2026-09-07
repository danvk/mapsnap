/**
 * A volume's key-map sheets and the sidecars the viewer can use for each.
 *
 * Key maps are `raw/<stem>.keymap.json`. Siblings: `<stem>.regions.panels.json`
 * (region view), `<stem>.georef.json` (the key map's own georeference, whose
 * `corners` place the sheet on the map) and `<stem>.roadprob.png` (the key
 * map's P(road) map, #211). The corners are what the key-map underlay draws
 * the sheet, or its P(road) map, with.
 */
import { readdir, readFile } from 'fs/promises';
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

/**
 * Key-map sheets under `rawDir`, sorted by stem, with their sidecars.
 *
 * `serviceBaseUrl` is the IIIF image service root for the raw directory
 * (`.../iiif/<volume>/raw`); each key map's sheet and P(road) map are
 * addressed under it as absolute URLs, the way the annotation rewrite
 * addresses page tiles, so the viewer reaches the API server directly in
 * development too (the Vite proxy does not forward `/iiif`).
 *
 * Empty when there is no raw/ directory: the volume has no key maps.
 */
export async function keymapInfos(
  rawDir: string,
  serviceBaseUrl: string,
): Promise<KeymapInfo[]> {
  let files: string[];
  try {
    files = await readdir(rawDir);
  } catch {
    return [];
  }
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
            await readFile(join(rawDir, `${stem}.georef.json`), 'utf8'),
          ),
        );
      } catch {
        corners = undefined;
      }
    }
    const image = ['jpg', 'png'].find((ext) => present.has(`${stem}.${ext}`));
    const hasRoadprob = present.has(`${stem}.roadprob.png`);
    infos.push({
      stem,
      hasRegions: present.has(`${stem}.regions.panels.json`),
      hasGeoref,
      hasRoadprob,
      ...(corners ? { corners } : {}),
      ...(image ? { imageService: `${serviceBaseUrl}/${stem}.${image}` } : {}),
      ...(hasRoadprob
        ? { roadprobService: `${serviceBaseUrl}/${stem}.roadprob.png` }
        : {}),
    });
  }
  return infos;
}
