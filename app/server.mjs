/**
 * Combined local server for the mapsnap debugger app.
 *
 * One process serves everything the browser UI needs against the repo's local
 * `data/` directory, so a single `npm run server` replaces the former separate
 * IIIF and key-map servers:
 *
 *   /iiif/...          IIIF Image API v3 for page JPEGs (via express-iiif)
 *   /iiif-api/...      volume list + rewritten Georeference AnnotationPages
 *   /api/...           key-map images + <stem>.labels.json truth sidecars
 *                      (fixed to <data>/keymaps)
 *   /notes-api/...     per-page free-text notes under
 *                      <data>/<volume>/artifacts/notes/<page>.json
 *   /mapsnap, /mapsnap/data   the production build and the data directory
 *
 * Usage:
 *   node server.mjs [data_dir] [port]
 *   npm run server                       # ../data on :8182
 *
 * In development `npm run dev` proxies /iiif, /iiif-api, /api and /notes-api
 * here; a production build (`npm run build`) is served standalone at /mapsnap.
 */

import { readdir, readFile, rm, stat, writeFile, mkdir } from 'fs/promises';
import { createRequire } from 'module';
import { dirname, join, resolve } from 'path';
import express from 'express';
import {
  rewriteAnnotationPage,
  serviceUrlToPageKey,
} from './server/iiifAnnotations.ts';
import { normalizeIiifImageUrl } from './server/iiifSizeWorkaround.ts';
import { jpegDimensions } from './server/jpegDimensions.ts';
import {
  noteFilePath,
  noteTextOf,
  notesDir,
  pageFromNoteFilename,
} from './server/notes.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const dataDir = resolve(process.argv[2] ?? '../data');
const keymapsDir = join(dataDir, 'keymaps');
const port = parseInt(process.argv[3] ?? '8182', 10);

const app = express();
app.use(express.json({ limit: '10mb' }));

// Open CORS for all origins — required for browser-based IIIF viewers (e.g.
// Allmaps) fetching /iiif tiles. Harmless for the same-origin JSON APIs.
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, HEAD, PUT, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Accept, Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// Reject a path segment that could escape its base directory.
function isSafeName(name) {
  return (
    name !== '' &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..')
  );
}

// ---------------------------------------------------------------------------
// IIIF Image API + volume/annotation JSON API
// ---------------------------------------------------------------------------

// See iiifSizeWorkaround: express-iiif mangles size===region requests.
app.use('/iiif', (req, _res, next) => {
  req.url = normalizeIiifImageUrl(req.url);
  next();
});
app.use('/iiif', iiif({ imageDir: dataDir }));

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

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

// Page keys with a note sidecar under a volume dir, as a Set (empty on error).
async function volumeNotePages(volume) {
  try {
    const files = await readdir(notesDir(dataDir, volume));
    const pages = files
      .map(pageFromNoteFilename)
      .filter((page) => page !== null);
    return new Set(pages);
  } catch {
    return new Set();
  }
}

// List volume directories that have local page images and annotation files.
app.get('/iiif-api/volumes', async (_req, res) => {
  try {
    const entries = await readdir(dataDir, { withFileTypes: true });
    const volumes = [];
    for (const entry of entries.filter((e) => e.isDirectory())) {
      const files = await readdir(join(dataDir, entry.name));
      const pageCount = files.filter((f) => PAGE_IMAGE_PATTERN.test(f)).length;
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

// ---------------------------------------------------------------------------
// Key-map truth API (fixed to <data>/keymaps)
// ---------------------------------------------------------------------------

// Map an image filename to its labels sidecar filename (foo.jpg → foo.labels.json).
function labelsFilename(imageName) {
  return imageName.replace(/\.[^.]+$/, '') + '.labels.json';
}

// List the available key map images, each with its current label count (if any).
app.get('/api/images', async (_req, res) => {
  try {
    const files = await readdir(keymapsDir);
    const images = files.filter((f) => /\.jpe?g$/i.test(f)).sort();
    const withMeta = await Promise.all(
      images.map(async (name) => {
        let labelCount = null;
        try {
          const text = await readFile(
            join(keymapsDir, labelsFilename(name)),
            'utf8',
          );
          labelCount = (JSON.parse(text).labels ?? []).length;
        } catch {
          // no sidecar yet
        }
        return { name, labelCount };
      }),
    );
    res.json({ images: withMeta });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// Serve a key map image file.
app.get('/api/keymaps/:name', (req, res) => {
  const { name } = req.params;
  if (!isSafeName(name)) return res.sendStatus(400);
  res.sendFile(join(keymapsDir, name));
});

// Read a key map image's labels sidecar, or report that none exists yet.
app.get('/api/labels/:name', async (req, res) => {
  const { name } = req.params;
  if (!isSafeName(name)) return res.sendStatus(400);
  try {
    const text = await readFile(join(keymapsDir, labelsFilename(name)), 'utf8');
    res.json(JSON.parse(text));
  } catch {
    res.json({ exists: false });
  }
});

// Write a key map image's labels sidecar.
app.put('/api/labels/:name', async (req, res) => {
  const { name } = req.params;
  if (!isSafeName(name)) return res.sendStatus(400);
  try {
    const path = join(keymapsDir, labelsFilename(name));
    await writeFile(path, JSON.stringify(req.body, null, 2) + '\n');
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// ---------------------------------------------------------------------------
// Per-page notes API
// ---------------------------------------------------------------------------

// The page keys with a note under one volume, for the volume viewer's markers
// and tooltip. ?volume=<dir> → { notes: { "<page>": "<text>" } }.
app.get('/notes-api/notes', async (req, res) => {
  const volume = String(req.query.volume ?? '');
  if (!isSafeName(volume)) return res.status(400).json({ error: 'bad volume' });
  const notes = {};
  for (const page of await volumeNotePages(volume)) {
    const path = noteFilePath(dataDir, volume, page);
    if (!path) continue;
    try {
      const text = noteTextOf(JSON.parse(await readFile(path, 'utf8')));
      if (text !== null) notes[page] = text;
    } catch {
      // Unreadable/blank sidecar — skip it.
    }
  }
  res.json({ notes });
});

// One page's note. ?volume=<dir>&page=<key> → { note: string }.
app.get('/notes-api/note', async (req, res) => {
  const path = noteFilePath(
    dataDir,
    String(req.query.volume ?? ''),
    String(req.query.page ?? ''),
  );
  if (!path) return res.status(400).json({ error: 'bad volume/page' });
  try {
    const text = noteTextOf(JSON.parse(await readFile(path, 'utf8')));
    res.json({ note: text ?? '' });
  } catch {
    res.json({ note: '' });
  }
});

// Write (or, when blank, delete) one page's note. Body: { note: string }.
app.put('/notes-api/note', async (req, res) => {
  const volume = String(req.query.volume ?? '');
  const page = String(req.query.page ?? '');
  const path = noteFilePath(dataDir, volume, page);
  if (!path) return res.status(400).json({ error: 'bad volume/page' });
  const note = String(req.body?.note ?? '');
  try {
    if (note.trim() === '') {
      await rm(path, { force: true });
    } else {
      await mkdir(notesDir(dataDir, volume), { recursive: true });
      const body = { note, updated: new Date().toISOString() };
      await writeFile(path, JSON.stringify(body, null, 2) + '\n');
    }
    res.json({ ok: true, note: note.trim() === '' ? '' : note });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// ---------------------------------------------------------------------------
// Static: the data directory and the production build under /mapsnap
// ---------------------------------------------------------------------------

// Serve the data directory under the app base so `?files=data/...` deep links
// work when the production build is served from here (the Vite dev server's
// serveDataDir plugin fills this role in development).
app.use('/mapsnap/data', express.static(dataDir));
app.use('/mapsnap', express.static(resolve('dist')));

app.listen(port, () => {
  console.error(`mapsnap server running at http://localhost:${port}`);
  console.error(`  data:    ${dataDir}`);
  console.error(`  keymaps: ${keymapsDir}`);
  console.error(`  UI (after build): http://localhost:${port}/mapsnap/`);
});
