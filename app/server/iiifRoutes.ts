/**
 * IIIF image serving + the volume/annotation JSON API.
 *
 * `registerIiifImages` mounts the raw binary endpoints (the express-iiif image
 * service under `/iiif`); `registerIiifApi` registers the typed JSON endpoints
 * (`/iiif-api/*`) on the shared crosswalk router.
 */

import { readdir, readFile, stat } from 'fs/promises';
import { createRequire } from 'module';
import { dirname, join } from 'path';
import type { Express } from 'express';
import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import {
  imageStemsByLowercase,
  rewriteAnnotationPage,
  serviceUrlToPageKey,
  type AnnotationFileInfo,
  type GeorefAnnotationPage,
  type VolumeInfo,
} from './iiifAnnotations.ts';
import { jpegDimensions } from './jpegDimensions.ts';
import {
  parseCompareFooter,
  parseCompareTxt,
  parseLandByPage,
  parseMissingTruthKeys,
} from './compareTxt.ts';
import { findVolumes, volumePages } from './adjacencyTruth.ts';
import { runArtifactDir, runArtifactStems } from './runArtifacts.ts';
import { withTiles } from './iiifAnnotations.ts';
import { isSafeSegment, isSafeVolume } from './volumePaths.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// A failed-georef sidecar name -> [full, stem, kind], e.g.
// "p1452.georef-nofit.json" -> ["…", "p1452", "nofit"].
const FAILED_GEOREF_PATTERN = /^(.+)\.georef-([a-z0-9]+)\.json$/i;

// Read a *.iiif.json if it is a georeference AnnotationPage, else null.
/**
 * Item counts for annotation files, keyed by path and invalidated by mtime.
 *
 * The volume list parses every `*.iiif.json` under data/ purely to show an item
 * count beside each file: 633 files and 175 MB of JSON at the time of writing,
 * about half a second, on every request. Counts cannot change without the file
 * changing, so a stat is enough to reuse one (#288).
 */
const itemCountCache = new Map<
  string,
  { mtimeMs: number; itemCount: number }
>();

async function annotationItemCount(
  path: string,
  mtimeMs: number,
): Promise<number | null> {
  const hit = itemCountCache.get(path);
  if (hit && hit.mtimeMs === mtimeMs) return hit.itemCount;
  const page = await readAnnotationPage(path);
  if (!page) return null;
  itemCountCache.set(path, { mtimeMs, itemCount: page.items.length });
  return page.items.length;
}

async function readAnnotationPage(
  path: string,
): Promise<GeorefAnnotationPage | null> {
  try {
    const data = JSON.parse(await readFile(path, 'utf8'));
    return data?.type === 'AnnotationPage' && Array.isArray(data.items)
      ? data
      : null;
  } catch {
    return null;
  }
}

// Page-image dimensions keyed by absolute path, invalidated by file mtime.
const dimensionsCache = new Map<
  string,
  { mtimeMs: number; dims: { width: number; height: number } }
>();

async function cachedJpegDimensions(
  path: string,
): Promise<{ width: number; height: number }> {
  const { mtimeMs } = await stat(path);
  const cached = dimensionsCache.get(path);
  if (cached && cached.mtimeMs === mtimeMs) return cached.dims;
  const dims = jpegDimensions(path);
  dimensionsCache.set(path, { mtimeMs, dims });
  return dims;
}

/**
 * Mount the raw IIIF image service (express-iiif) under `/iiif`.
 *
 * info.json responses are passed through {@link withTiles} first; see there
 * for why an advertised tileset is load-bearing for the map viewer.
 */
export function registerIiifImages(app: Express, dataDir: string): void {
  app.use('/iiif', (request, response, next) => {
    if (!request.path.endsWith('/info.json')) return next();
    const json = response.json.bind(response);
    response.json = (body: unknown) =>
      json(
        body && typeof body === 'object'
          ? withTiles(body as Record<string, unknown>)
          : body,
      );
    next();
  });
  app.use('/iiif', iiif({ imageDir: dataDir }));
}

/** Register the typed volume/annotation JSON API (`/iiif-api/*`). */
export function registerIiifApi(
  router: TypedRouter<API>,
  dataDir: string,
): void {
  // Volume directories that have local page images and annotation files.
  //
  // findVolumes descends into a multi-volume atlas, so `brooklyn_1904-1908/vol13`
  // is listed under that name rather than being missed because its parent holds no
  // page images (#228). It stops at the first page-bearing directory on a branch,
  // so a volume's own `raw/` is never listed as a volume.
  router.get('/iiif-api/volumes', async () => {
    // Every volume, and every annotation within it, is read CONCURRENTLY.
    // Serially this walked ~18 volumes x several annotation files each,
    // parsing every one in full for its item count, and took long enough that
    // the volume picker arrived after the map had drawn (#288).
    const names = await findVolumes(dataDir);
    const built = await Promise.all(
      names.map(async (name): Promise<VolumeInfo | null> => {
        const files = await readdir(join(dataDir, name));
        const pageCount = files.filter((f) =>
          PAGE_IMAGE_PATTERN.test(f),
        ).length;
        if (pageCount === 0) return null;
        const annotations = (
          await Promise.all(
            files
              .filter((f) => f.endsWith('.iiif.json'))
              .map(async (file): Promise<AnnotationFileInfo | null> => {
                const path = join(dataDir, name, file);
                const info = await stat(path);
                const itemCount = await annotationItemCount(path, info.mtimeMs);
                return itemCount === null
                  ? null
                  : {
                      name: file,
                      modifiedMs: Math.round(info.mtimeMs),
                      itemCount,
                    };
              }),
          )
        ).filter((a): a is AnnotationFileInfo => a !== null);
        if (annotations.length === 0) return null;
        annotations.sort((a, b) => b.modifiedMs - a.modifiedMs);
        return { name, pageCount, annotations };
      }),
    );
    const volumes = built.filter((v): v is VolumeInfo => v !== null);
    volumes.sort((a, b) => a.name.localeCompare(b.name));
    return { volumes };
  });

  // Serve an AnnotationPage rewritten to target this server's /iiif endpoint.
  // The path is repo-root-relative like the app's ?files= param, so a leading
  // "data/" is tolerated (dataDir already points at the data directory).
  router.get('/iiif-api/annotation', async (_params, request) => {
    const rawPath = request.query.path;
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const annotationPath = join(dataDir, relativePath);
    const page = await readAnnotationPage(annotationPath);
    if (!page) {
      throw new HTTPError(
        404,
        `not found or not an AnnotationPage: ${rawPath}`,
      );
    }
    const volumeDir = dirname(annotationPath);
    const stems = await imageStemsByLowercase(volumeDir);
    const localPages = new Map<string, { width: number; height: number }>();
    for (const item of page.items) {
      // Same (url, label) pair rewriteAnnotationPage will use: the label
      // carries the split-panel variant and, for volumes that link no image
      // service, the page number itself. Deriving the key differently here
      // keys localPages under names the rewrite never looks up, and every
      // page reports "missing-image".
      const derived = serviceUrlToPageKey(
        item?.target?.source?.id,
        String(item?.label ?? item?.id ?? ''),
        String(item?.id ?? ''),
      );
      // A split panel is georeferenced against its parent sheet, so the image
      // to measure and serve is the parent (see rewriteAnnotationPage).
      const parent = derived?.replace(/__\d+$/, '');
      if (!parent) continue;
      // Volumes disagree about the case of a lettered suffix -- Chicago's
      // 0103W is p103w.jpg on disk, Asheville's 0033A is p33A.jpg -- so the
      // derived key's case is a guess. Resolve it against the directory and
      // use the real stem, or every link the viewer builds from it 404s on a
      // case-sensitive filesystem (and reads the wrong name on a forgiving one).
      const imageKey = stems.get(parent.toLowerCase()) ?? parent;
      if (localPages.has(imageKey)) continue;
      try {
        const dims = await cachedJpegDimensions(
          join(volumeDir, `${imageKey}.jpg`),
        );
        localPages.set(imageKey, dims);
      } catch {
        // No local image for this page; rewriteAnnotationPage reports it.
      }
    }
    const volumePath = parts.slice(0, -1).join('/');
    const serviceBaseUrl = `${request.protocol}://${request.get('host')}/iiif/${volumePath}`;
    return rewriteAnnotationPage(page, localPages, serviceBaseUrl, stems);
  });

  // Where a run's own per-page sidecars live, so the viewer can link to the files
  // that produced the annotation being looked at rather than to whatever the last
  // run happened to leave at the top level.
  router.get('/iiif-api/run-artifacts', async (_params, request) => {
    const rawPath = request.query.path;
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const artifactDir = runArtifactDir(relativePath);
    if (!artifactDir) return { dir: null, stems: [] };

    let files: string[];
    try {
      files = await readdir(join(dataDir, artifactDir));
    } catch {
      // A run with no saved sidecars is the common case, not an error.
      return { dir: null, stems: [] };
    }
    const stems = runArtifactStems(files);
    return stems.length > 0
      ? { dir: `data/${artifactDir}`, stems }
      : { dir: null, stems: [] };
  });

  // Per-page truth comparison from the annotation's `mapsnap compare` sidecar table
  // (`<name>.txt` next to `<name>.iiif.json`). Empty when there is no sidecar.
  router.get('/iiif-api/compare', async (_params, request) => {
    const rawPath = request.query.path;
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const txtPath = join(
      dataDir,
      relativePath.replace(/\.iiif\.json$/, '.txt'),
    );
    try {
      const text = await readFile(txtPath, 'utf8');
      return {
        pages: parseCompareTxt(text),
        missing: parseMissingTruthKeys(text),
        ...((land) => (land ? { landKm2ByPage: land } : {}))(
          parseLandByPage(text),
        ),
        footer: parseCompareFooter(text),
      };
    } catch {
      return { pages: [], missing: [], footer: '' };
    }
  });

  // A volume's adjacency.json (per-page sheet-number claims + the mutual-edge graph),
  // for the viewer's adjacency overlay. Null when the volume has no adjacency data.
  router.get('/iiif-api/adjacency', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    try {
      const text = await readFile(
        join(dataDir, volume, 'adjacency.json'),
        'utf8',
      );
      return { adjacency: JSON.parse(text) };
    } catch {
      return { adjacency: null };
    }
  });

  // The boundary of the OSM relation this volume's streets were downloaded from
  // (data/<volume>/r<id>.json, saved by the coverage sweep). The viewer draws it
  // so a page whose ground falls OUTSIDE the download is visible as such: those
  // pages' streets are missing from the vocabulary entirely, which is why
  // richmond p383 and fargo's Moorhead sheets cannot be fit at all.
  router.get('/iiif-api/osm-relation', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    try {
      const dir = join(dataDir, volume);
      // Which relation the streets came from is recorded in the volume's own
      // manifest, so a leftover r<id>.json from an earlier download is ignored
      // rather than drawing a boundary the current streets did not come from.
      const manifest = JSON.parse(
        await readFile(join(dir, 'mapsnap.json'), 'utf8'),
      );
      const name: unknown = manifest?.params?.relation;
      if (typeof name !== 'string' || !/^r\d+$/.test(name)) {
        return { relation: null };
      }
      const doc = JSON.parse(await readFile(join(dir, `${name}.json`), 'utf8'));
      const element = doc.elements?.[0];
      if (!element) return { relation: null };
      const ways = (element.members ?? [])
        .filter((m: any) => m.type === 'way' && Array.isArray(m.geometry))
        .map((m: any) =>
          m.geometry.map((p: any) => [p.lon, p.lat] as [number, number]),
        )
        .filter((w: unknown[]) => w.length >= 2);
      return {
        relation: {
          id: name,
          name: element.tags?.name ?? null,
          ways,
        },
      };
    } catch {
      return { relation: null };
    }
  });

  // A volume's page files: every page-image stem, plus the ones with a failed-georef
  // sidecar and that sidecar's kind, so the viewer can link an un-georeferenced page to
  // its georef-<kind>.json file and — for a volume with no truth annotation — work out
  // which pages went unplaced at all. ?volume=<dir> →
  // { pages: ["p1", …], failed: { "p1452": "nofit", "p1427": "misscale" } }.
  router.get('/iiif-api/failed-georefs', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    let files: string[];
    try {
      files = await readdir(join(dataDir, volume));
    } catch {
      throw new HTTPError(404, `no such volume: ${volume}`);
    }
    const failed: Record<string, string> = {};
    for (const file of files) {
      const match = file.match(FAILED_GEOREF_PATTERN);
      // First kind wins if a page somehow has more than one failed sidecar.
      if (match && match[1] && match[2] && !(match[1] in failed)) {
        failed[match[1]] = match[2].toLowerCase();
      }
    }
    // volumePages drops a split sheet in favour of its panels, so a sheet whose panels
    // all fitted is not reported as an unplaced page.
    return { failed, pages: await volumePages(dataDir, volume) };
  });

  // A volume's key-map sheets and which visualization sidecars each has, so the viewer can link
  // to them. Key maps are `raw/<stem>.keymap.json`; siblings <stem>.regions.panels.json and
  // <stem>.georef.json are the region and georef views. ?volume=<dir> → { keymaps: [...] }.
  router.get('/iiif-api/keymaps', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    let files: string[];
    try {
      files = await readdir(join(dataDir, volume, 'raw'));
    } catch {
      return { keymaps: [] }; // no raw/ directory: volume has no key maps
    }
    const present = new Set(files);
    const keymaps = files
      .filter((file) => file.endsWith('.keymap.json'))
      .map((file) => file.slice(0, -'.keymap.json'.length))
      .sort()
      .map((stem) => ({
        stem,
        hasRegions: present.has(`${stem}.regions.panels.json`),
        hasGeoref: present.has(`${stem}.georef.json`),
      }));
    return { keymaps };
  });
}
