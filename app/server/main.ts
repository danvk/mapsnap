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
 *   npm run server                          # ../data on :8182
 *
 * In development `npm run dev` proxies /iiif, /iiif-api, /api and /notes-api
 * here; a production build (`npm run build`) is served standalone at /mapsnap.
 */

import { join, resolve } from 'path';
import express from 'express';
import { TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import { registerIiifApi, registerIiifImages } from './iiifRoutes.ts';
import { registerKeymapApi, registerKeymapImages } from './keymapRoutes.ts';
import { registerNotesApi } from './notesRoutes.ts';

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

// Binary endpoints (raw Express): the IIIF image service and key-map JPEGs.
// Registered before the typed router so their more specific paths win.
registerIiifImages(app, dataDir);
registerKeymapImages(app, keymapsDir);

// The typed JSON API (crosswalk), defined by the API interface in ./api.
const router = new TypedRouter<API>(app);
registerIiifApi(router, dataDir);
registerKeymapApi(router, keymapsDir);
registerNotesApi(router, dataDir);

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
