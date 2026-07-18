/**
 * IIIF Image API + volume/annotation JSON API, mounted on a shared Express app.
 *
 * Serves page JPEGs under the data directory as a IIIF Image API v3 endpoint
 * (via express-iiif) plus the debugger's volume/annotation JSON API:
 *   GET /iiif/<vol>/<file>.jpg/info.json  →  the image, IIIF-style
 *   GET /iiif-api/volumes                 →  volume dirs with pages + annotations
 *   GET /iiif-api/annotation?path=...     →  an AnnotationPage rewritten to target
 *                                            this server's /iiif endpoint
 *
 * registerIiifRoutes(app, { dataDir }) wires these onto the app; server.mjs owns
 * the process (shared middleware, static mounts, listen).
 */

import { readdir, readFile, stat } from 'fs/promises';
import { createRequire } from 'module';
import { dirname, join } from 'path';
import {
  rewriteAnnotationPage,
  serviceUrlToPageKey,
} from './server/iiifAnnotations.ts';
import { normalizeIiifImageUrl } from './server/iiifSizeWorkaround.ts';
import { jpegDimensions } from './server/jpegDimensions.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// Reject a path segment that could escape the data directory.
function isSafeName(name) {
  return (
    name !== '' &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..')
  );
}

// Read a *.iiif.json if it is a georeference AnnotationPage, else null.
async function readAnnotationPage(path) {
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
const dimensionsCache = new Map();

async function cachedJpegDimensions(path) {
  const { mtimeMs } = await stat(path);
  const cached = dimensionsCache.get(path);
  if (cached && cached.mtimeMs === mtimeMs) return cached.dims;
  const dims = jpegDimensions(path);
  dimensionsCache.set(path, { mtimeMs, dims });
  return dims;
}

/** Mount the IIIF image endpoint and volume/annotation JSON API on ``app``. */
export function registerIiifRoutes(app, { dataDir }) {
  // See iiifSizeWorkaround: express-iiif mangles size===region requests.
  app.use('/iiif', (req, _res, next) => {
    req.url = normalizeIiifImageUrl(req.url);
    next();
  });
  app.use('/iiif', iiif({ imageDir: dataDir }));

  // List volume directories that have local page images and annotation files.
  app.get('/iiif-api/volumes', async (_req, res) => {
    try {
      const entries = await readdir(dataDir, { withFileTypes: true });
      const volumes = [];
      for (const entry of entries.filter((e) => e.isDirectory())) {
        const files = await readdir(join(dataDir, entry.name));
        const pageCount = files.filter((f) =>
          PAGE_IMAGE_PATTERN.test(f),
        ).length;
        if (pageCount === 0) continue;
        const annotations = [];
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
      res.json({ volumes });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // Serve an AnnotationPage rewritten to target this server's /iiif endpoint.
  // The path is repo-root-relative like the app's ?files= param, so a leading
  // "data/" is tolerated (dataDir already points at the data directory).
  app.get('/iiif-api/annotation', async (req, res) => {
    try {
      const rawPath = String(req.query.path ?? '');
      const relativePath = rawPath.replace(/^data\//, '');
      const parts = relativePath.split('/');
      if (
        !relativePath.endsWith('.iiif.json') ||
        parts.length < 2 ||
        !parts.every(isSafeName)
      ) {
        return res.status(400).json({ error: `invalid path: ${rawPath}` });
      }
      const annotationPath = join(dataDir, relativePath);
      const page = await readAnnotationPage(annotationPath);
      if (!page) {
        return res
          .status(404)
          .json({ error: `not found or not an AnnotationPage: ${rawPath}` });
      }
      const volumeDir = dirname(annotationPath);
      const localPages = new Map();
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
      const serviceBaseUrl = `${req.protocol}://${req.get('host')}/iiif/${volumePath}`;
      res.json(rewriteAnnotationPage(page, localPages, serviceBaseUrl));
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });
}
