/**
 * Key-map truth API (fixed to <data>/keymaps).
 *
 * `registerKeymapImages` mounts the raw key-map JPEG endpoint; `registerKeymapApi`
 * registers the typed JSON endpoints (image list + label sidecars) on the shared
 * crosswalk router.
 */

import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';
import type { Express } from 'express';
import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API, KeymapImagesResponse } from './api.ts';
import type { ImageInfo } from '../src/keymap/types.ts';

// Reject a name that could escape the keymaps directory. A `:name` route param
// is one path segment but can still carry an encoded slash, so this guards the
// label sidecar path against traversal.
function isSafeName(name: string): boolean {
  return !name.includes('/') && !name.includes('\\') && !name.includes('..');
}

// Map an image filename to its labels sidecar filename (foo.jpg → foo.labels.json).
function labelsFilename(imageName: string): string {
  return imageName.replace(/\.[^.]+$/, '') + '.labels.json';
}

/** Mount the raw key-map image endpoint under `/api/keymaps/:name`. */
export function registerKeymapImages(app: Express, keymapsDir: string): void {
  app.get('/api/keymaps/:name', (req, res) => {
    const { name } = req.params;
    if (!isSafeName(name)) return res.sendStatus(400);
    res.sendFile(join(keymapsDir, name));
  });
}

/** Register the typed key-map image-list and label-sidecar API (`/api/*`). */
export function registerKeymapApi(
  router: TypedRouter<API>,
  keymapsDir: string,
): void {
  // List the available key map images, each with its current label count (if any).
  router.get('/api/images', async (): Promise<KeymapImagesResponse> => {
    const files = await readdir(keymapsDir);
    const images = files.filter((f) => /\.jpe?g$/i.test(f)).sort();
    const withMeta: ImageInfo[] = await Promise.all(
      images.map(async (name) => {
        let labelCount: number | null = null;
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
    return { images: withMeta };
  });

  // Read a key map image's labels sidecar, or report that none exists yet.
  router.get('/api/labels/:name', async ({ name }) => {
    if (!isSafeName(name)) throw new HTTPError(400, 'bad name');
    try {
      const text = await readFile(
        join(keymapsDir, labelsFilename(name)),
        'utf8',
      );
      return JSON.parse(text);
    } catch {
      return { exists: false };
    }
  });

  // Write a key map image's labels sidecar.
  router.put('/api/labels/:name', async ({ name }, body) => {
    if (!isSafeName(name)) throw new HTTPError(400, 'bad name');
    const path = join(keymapsDir, labelsFilename(name));
    await writeFile(path, JSON.stringify(body, null, 2) + '\n');
    return { ok: true };
  });
}
