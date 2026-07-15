/**
 * Local IIIF Image API v3 server for debugging.
 *
 * Serves images from a local directory as a IIIF Image API v3 endpoint,
 * enabling fast local testing without network round-trips to the LOC or Lambda.
 *
 * Usage:
 *   node iiif-server.mjs <image_dir> [port]
 *   npm run iiif -- ../data/brooklyn_1904-1908 8182
 *
 * The identifier in the IIIF URL maps directly to a file under <image_dir>:
 *   GET /iiif/vol1/p12.raw.jpg/info.json  →  <image_dir>/vol1/p12.raw.jpg
 *
 * Also serves a JSON API for the debugger app's volume viewer:
 *   GET /iiif-api/volumes           →  volume dirs with pages + annotations
 *   GET /iiif-api/annotation?path=data/<vol>/<file>.iiif.json
 *       →  that AnnotationPage, rewritten to target this server's /iiif
 *          endpoint with coordinates rescaled to the local image dimensions
 *
 * When a production build exists, it is served at /mapsnap, so
 * `npm run build && npm run iiif` works standalone.
 */

import { readdir, readFile, stat } from 'fs/promises';
import { createRequire } from 'module';
import { dirname, join, resolve } from 'path';
import {
  rewriteAnnotationPage,
  serviceUrlToPageKey,
} from './server/iiifAnnotations.ts';
import { jpegDimensions } from './server/jpegDimensions.ts';

const require = createRequire(import.meta.url);
const express = require('express');
const iiif = require('express-iiif').default;

const imageDir = resolve(process.argv[2] ?? '../data');
const port = parseInt(process.argv[3] ?? '8182', 10);

const app = express();

// Open CORS for all origins — required for browser-based IIIF viewers (e.g. Allmaps).
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Accept, Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

app.use('/iiif', iiif({ imageDir }));

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// Reject path segments that could escape the image directory.
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

// List volume directories that have local page images and annotation files.
app.get('/iiif-api/volumes', async (_req, res) => {
  try {
    const entries = await readdir(imageDir, { withFileTypes: true });
    const volumes = [];
    for (const entry of entries.filter((e) => e.isDirectory())) {
      const files = await readdir(join(imageDir, entry.name));
      const pageCount = files.filter((f) => PAGE_IMAGE_PATTERN.test(f)).length;
      if (pageCount === 0) continue;
      const annotations = [];
      for (const file of files.filter((f) => f.endsWith('.iiif.json'))) {
        const path = join(imageDir, entry.name, file);
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
// "data/" is tolerated (imageDir already points at the data directory).
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
    const annotationPath = join(imageDir, relativePath);
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

// Serve a production build, if one exists, under the app's base path.
app.use('/mapsnap', express.static(resolve('dist')));

app.listen(port, () => {
  console.error(`IIIF server running at http://localhost:${port}/iiif`);
  console.error(`Serving images from: ${imageDir}`);
  console.error(
    `Example: http://localhost:${port}/iiif/vol1/p1.raw.jpg/info.json`,
  );
});
