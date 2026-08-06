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
import { isSafeSegment, isSafeVolume } from './volumePaths.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// A failed-georef sidecar name -> [full, stem, kind], e.g.
// "p1452.georef-nofit.json" -> ["…", "p1452", "nofit"].
const FAILED_GEOREF_PATTERN = /^(.+)\.georef-([a-z0-9]+)\.json$/i;

// Read a *.iiif.json if it is a georeference AnnotationPage, else null.
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

/** Mount the raw IIIF image service (express-iiif) under `/iiif`. */
export function registerIiifImages(app: Express, dataDir: string): void {
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
    const volumes: VolumeInfo[] = [];
    for (const name of await findVolumes(dataDir)) {
      const files = await readdir(join(dataDir, name));
      const pageCount = files.filter((f) => PAGE_IMAGE_PATTERN.test(f)).length;
      if (pageCount === 0) continue;
      const annotations: AnnotationFileInfo[] = [];
      for (const file of files.filter((f) => f.endsWith('.iiif.json'))) {
        const path = join(dataDir, name, file);
        const page = await readAnnotationPage(path);
        if (!page) continue;
        const { mtimeMs } = await stat(path);
        annotations.push({
          name: file,
          modifiedMs: Math.round(mtimeMs),
          itemCount: page.items.length,
        });
      }
      if (annotations.length === 0) continue;
      annotations.sort((a, b) => b.modifiedMs - a.modifiedMs);
      volumes.push({ name, pageCount, annotations });
    }
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
      const derived = serviceUrlToPageKey(item?.target?.source?.id);
      if (!derived) continue;
      // Volumes disagree about the case of a lettered suffix -- Chicago's
      // 0103W is p103w.jpg on disk, Asheville's 0033A is p33A.jpg -- so the
      // derived key's case is a guess. Resolve it against the directory and
      // use the real stem, or every link the viewer builds from it 404s on a
      // case-sensitive filesystem (and reads the wrong name on a forgiving one).
      const pageKey = stems.get(derived.toLowerCase()) ?? derived;
      if (localPages.has(pageKey)) continue;
      try {
        const dims = await cachedJpegDimensions(
          join(volumeDir, `${pageKey}.jpg`),
        );
        localPages.set(pageKey, dims);
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
