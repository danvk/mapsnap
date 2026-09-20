/**
 * Combined local server for the mapsnap debugger app.
 *
 * One process serves everything the browser UI needs against the repo's local
 * `data/` directory, so a single `npm run server` replaces the former separate
 * IIIF and key-map servers. The JSON API is defined once in ./api and served
 * type-safely with crosswalk's TypedRouter; the binary image endpoints and the
 * static build are plain Express middleware.
 *
 * Usage:
 *   node server/main.ts [data_dir] [port]
 *   npm run server                          # ../data on :8182, restarts on edit
 *   npm run server:once                     # same, without the watcher
 *
 * `npm run server` runs under `node --watch-path=./server`, so editing anything
 * in this directory restarts the process -- the client half of the app has
 * always hot-reloaded and the server half not doing so was a standing papercut
 * (#203). The watch is scoped to ./server rather than the whole module graph so
 * that editing a page under src/ does not bounce the API, and it covers files
 * that are not imported (a new route module, say) which a graph-based watch
 * would miss. `server:once` is the unwatched form for anything scripted.
 *
 * In development `npm run dev` proxies /iiif, /iiif-api, /api and /notes-api
 * here; a production build (`npm run build`) is served standalone at /mapsnap.
 */

import { homedir } from 'os';
import { join, resolve } from 'path';
import express from 'express';
import { TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import { registerIiifApi, registerIiifImages } from './iiifRoutes.ts';
import { registerS3IiifImages } from './s3Routes.ts';
import { registerAdjacencyTruthApi } from './adjacencyRoutes.ts';
import { registerKeymapApi, registerKeymapImages } from './keymapRoutes.ts';
import { registerNotesApi } from './notesRoutes.ts';

const dataDir = resolve(process.argv[2] ?? '../data');
// Mirror scans fetched for `?iiif=s3://…`, kept out of the repo and out of
// data/ so nothing walks them as a volume. Persistent on purpose: a debugging
// session revisits the same pages, and refetching a 120-page volume each time
// is what made loc.gov unusable in the first place.
const s3CacheDir =
  process.env.MAPSNAP_S3_CACHE ?? join(homedir(), '.cache', 'mapsnap', 's3');
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

// Binary endpoints (raw Express): the IIIF image service and key-map images.
// Registered before the typed router so their more specific paths win.
registerIiifImages(app, dataDir);
registerS3IiifImages(app, s3CacheDir);
registerKeymapImages(app, dataDir);

// The typed JSON API (crosswalk), defined by the API interface in ./api.
const router = new TypedRouter<API>(app);
registerIiifApi(router, dataDir, s3CacheDir);
registerKeymapApi(router, dataDir);
registerAdjacencyTruthApi(router, dataDir);
registerNotesApi(router, dataDir);

// Serve the data directory under the app base so `?files=data/...` deep links
// work when the production build is served from here (the Vite dev server's
// serveDataDir plugin fills this role in development).
app.use('/mapsnap/data', express.static(dataDir));
app.use('/mapsnap', express.static(resolve('dist')));

app.listen(port, () => {
  console.error(`mapsnap server running at http://localhost:${port}`);
  console.error(`  data:    ${dataDir}`);
  console.error(`  s3 cache: ${s3CacheDir}`);
  console.error(`  UI (after build): http://localhost:${port}/mapsnap/`);
});
