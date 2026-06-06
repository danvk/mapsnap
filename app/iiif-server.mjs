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
 */

import { createRequire } from 'module';
import { resolve } from 'path';

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

app.listen(port, () => {
  console.error(`IIIF server running at http://localhost:${port}/iiif`);
  console.error(`Serving images from: ${imageDir}`);
  console.error(
    `Example: http://localhost:${port}/iiif/vol1/p1.raw.jpg/info.json`,
  );
});
