/**
 * Combined local server for the mapsnap debugger app.
 *
 * One process serves everything the browser UI needs against the repo's local
 * `data/` directory, so a single `npm run server` replaces the former separate
 * IIIF and key-map servers. Each API keeps its own file and registers its own
 * routes on the shared app:
 *
 *   /iiif, /iiif-api   iiif-server.mjs    page JPEGs + rewritten annotations
 *   /api               keymap-server.mjs  key maps + labels (fixed to <data>/keymaps)
 *   /notes-api         notes-server.mjs   per-page notes under artifacts/notes/
 *   /mapsnap[/data]    (here)             the production build and the data directory
 *
 * Usage:
 *   node server.mjs [data_dir] [port]
 *   npm run server                       # ../data on :8182
 *
 * In development `npm run dev` proxies /iiif, /iiif-api, /api and /notes-api
 * here; a production build (`npm run build`) is served standalone at /mapsnap.
 */

import { join, resolve } from 'path';
import express from 'express';
import { registerIiifRoutes } from './iiif-server.mjs';
import { registerKeymapRoutes } from './keymap-server.mjs';
import { registerNotesRoutes } from './notes-server.mjs';

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

registerIiifRoutes(app, { dataDir });
registerKeymapRoutes(app, { keymapsDir });
registerNotesRoutes(app, { dataDir });

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
