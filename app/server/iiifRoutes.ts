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
  rewriteAnnotationPage,
  serviceUrlToPageKey,
  type AnnotationFileInfo,
  type GeorefAnnotationPage,
  type VolumeInfo,
} from './iiifAnnotations.ts';
import { normalizeIiifImageUrl } from './iiifSizeWorkaround.ts';
import { jpegDimensions } from './jpegDimensions.ts';
import { parseCompareFooter, parseCompareTxt } from './compareTxt.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// A failed-georef sidecar name -> [full, stem, kind], e.g.
// "p1452.georef-nofit.json" -> ["…", "p1452", "nofit"].
const FAILED_GEOREF_PATTERN = /^(.+)\.georef-([a-z0-9]+)\.json$/i;

// Reject a path segment that could escape the data directory.
function isSafeName(name: string): boolean {
  return (
    name !== '' &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..')
  );
}

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
  // See iiifSizeWorkaround: express-iiif mangles size===region requests.
  app.use('/iiif', (req, _res, next) => {
    req.url = normalizeIiifImageUrl(req.url);
    next();
  });
  app.use('/iiif', iiif({ imageDir: dataDir }));
}

/** Register the typed volume/annotation JSON API (`/iiif-api/*`). */
export function registerIiifApi(
  router: TypedRouter<API>,
  dataDir: string,
): void {
  // List volume directories that have local page images and annotation files.
  router.get('/iiif-api/volumes', async () => {
    const entries = await readdir(dataDir, { withFileTypes: true });
    const volumes: VolumeInfo[] = [];
    for (const entry of entries.filter((e) => e.isDirectory())) {
      const files = await readdir(join(dataDir, entry.name));
      const pageCount = files.filter((f) => PAGE_IMAGE_PATTERN.test(f)).length;
      if (pageCount === 0) continue;
      const annotations: AnnotationFileInfo[] = [];
      for (const file of files.filter((f) => f.endsWith('.iiif.json'))) {
        const path = join(dataDir, entry.name, file);
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
      volumes.push({ name: entry.name, pageCount, annotations });
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
      !parts.every(isSafeName)
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
    const localPages = new Map<string, { width: number; height: number }>();
    for (const item of page.items) {
      const pageKey = serviceUrlToPageKey(item?.target?.source?.id);
      if (!pageKey || localPages.has(pageKey)) continue;
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
    return rewriteAnnotationPage(page, localPages, serviceBaseUrl);
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
      !parts.every(isSafeName)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const txtPath = join(
      dataDir,
      relativePath.replace(/\.iiif\.json$/, '.txt'),
    );
    try {
      const text = await readFile(txtPath, 'utf8');
      return { pages: parseCompareTxt(text), footer: parseCompareFooter(text) };
    } catch {
      return { pages: [], footer: '' };
    }
  });

  // A volume's adjacency.json (per-page sheet-number claims + the mutual-edge graph),
  // for the viewer's adjacency overlay. Null when the volume has no adjacency data.
  router.get('/iiif-api/adjacency', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeName(volume)) {
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

  // Page stems with a failed-georef sidecar in a volume, and each one's kind, so
  // the viewer can link an un-georeferenced page to its georef-<kind>.json file.
  // ?volume=<dir> → { failed: { "p1452": "nofit", "p1427": "misscale" } }.
  router.get('/iiif-api/failed-georefs', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeName(volume)) {
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
    return { failed };
  });

  // A volume's key-map sheets and which visualization sidecars each has, so the viewer can link
  // to them. Key maps are `raw/<stem>.keymap.json`; siblings <stem>.regions.panels.json and
  // <stem>.georef.json are the region and georef views. ?volume=<dir> → { keymaps: [...] }.
  router.get('/iiif-api/keymaps', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeName(volume)) {
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
