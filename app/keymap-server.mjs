/**
 * Truth-data server for key map images.
 *
 * Serves the key map JPEGs under a directory plus a JSON API for reading and
 * writing per-image <stem>.labels.json sidecars, so the browser UI can persist
 * truth data to the local filesystem.
 *
 * Usage:
 *   node keymap-server.mjs [image_dir] [port]
 *   npm run keymap -- ../data/keymaps 8183
 *
 * In development, run `npm run dev` alongside this and open the Vite URL; Vite
 * proxies /api to this server. When a production build exists, this server also
 * serves it at /mapsnap, so `npm run build && npm run keymap` works standalone.
 */

import express from 'express';
import { readdir, readFile, writeFile } from 'fs/promises';
import { join, resolve } from 'path';

const keymapsDir = resolve(process.argv[2] ?? '../data/keymaps');
const port = parseInt(process.argv[3] ?? '8183', 10);

const app = express();
app.use(express.json({ limit: '10mb' }));

// Map an image filename to its labels sidecar filename (foo.jpg → foo.labels.json).
function labelsFilename(imageName) {
  return imageName.replace(/\.[^.]+$/, '') + '.labels.json';
}

// Reject names that could escape the keymaps directory.
function isSafeName(name) {
  return !name.includes('/') && !name.includes('\\') && !name.includes('..');
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

// Serve an image file.
app.get('/api/keymaps/:name', (req, res) => {
  const { name } = req.params;
  if (!isSafeName(name)) return res.sendStatus(400);
  res.sendFile(join(keymapsDir, name));
});

// Read an image's labels sidecar, or report that none exists yet.
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

// Write an image's labels sidecar.
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

// Serve a production build, if one exists, under the app's base path.
app.use('/mapsnap', express.static(resolve('dist')));

app.listen(port, () => {
  console.error(`Keymap server running at http://localhost:${port}`);
  console.error(`Serving key maps from: ${keymapsDir}`);
  console.error(
    `UI (after build): http://localhost:${port}/mapsnap/keymap.html`,
  );
});
