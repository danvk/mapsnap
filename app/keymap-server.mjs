/**
 * Key-map truth API, mounted on a shared Express app.
 *
 * Serves the key map JPEGs under <data>/keymaps plus a JSON API for reading and
 * writing per-image <stem>.labels.json sidecars, so the labeler UI can persist
 * truth data to the local filesystem:
 *   GET /api/images            →  key map images with their label counts
 *   GET /api/keymaps/<name>    →  a key map image
 *   GET/PUT /api/labels/<name> →  that image's labels sidecar
 *
 * registerKeymapRoutes(app, { keymapsDir }) wires these onto the app; server.mjs
 * owns the process.
 */

import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

// Reject a name that could escape the keymaps directory.
function isSafeName(name) {
  return !name.includes('/') && !name.includes('\\') && !name.includes('..');
}

// Map an image filename to its labels sidecar filename (foo.jpg → foo.labels.json).
function labelsFilename(imageName) {
  return imageName.replace(/\.[^.]+$/, '') + '.labels.json';
}

/** Mount the key-map image and labels API on ``app``. */
export function registerKeymapRoutes(app, { keymapsDir }) {
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
      const text = await readFile(
        join(keymapsDir, labelsFilename(name)),
        'utf8',
      );
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
}
