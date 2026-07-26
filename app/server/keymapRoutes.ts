/**
 * Key-map truth API over the volumes' `raw/` directories.
 *
 * `registerKeymapImages` mounts the raw key-map image endpoint;
 * `registerKeymapApi` registers the typed JSON endpoints (page list + label
 * sidecars) on the shared crosswalk router. Pages are addressed by the ids
 * described in ./keymapPages — "queens_1950/vol2/p0" — which contain slashes,
 * so the JSON endpoints take the id as a query parameter and the image
 * endpoint takes it as a wildcard path.
 */

import { mkdir, readFile, writeFile } from 'fs/promises';
import { basename, dirname } from 'path';
import type { Express } from 'express';
import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API, KeymapImagesResponse } from './api.ts';
import {
  findKeymapPages,
  readLabelCount,
  resolveKeymapPage,
} from './keymapPages.ts';
import type { ImageInfo } from '../src/keymap/types.ts';

/**
 * Mount the raw key-map image endpoint under `/api/keymaps/<page id>`.
 *
 * The `(*)` makes `:id` greedy so it captures the slashes in a page id;
 * resolveKeymapPage is what keeps it from reaching outside the data directory.
 */
export function registerKeymapImages(app: Express, dataDir: string): void {
  app.get('/api/keymaps/:id(*)', async (req, res) => {
    const page = await resolveKeymapPage(dataDir, req.params.id);
    if (!page) return res.sendStatus(404);
    res.sendFile(page.imagePath);
  });
}

/** Register the typed key-map page-list and label-sidecar API (`/api/*`). */
export function registerKeymapApi(
  router: TypedRouter<API>,
  dataDir: string,
): void {
  // List the key-map pages of every volume, each with its current label count.
  router.get('/api/images', async (): Promise<KeymapImagesResponse> => {
    const pages = await findKeymapPages(dataDir);
    const images: ImageInfo[] = await Promise.all(
      pages.map(async (page) => ({
        name: page.id,
        labelCount: await readLabelCount(page.labelsPath),
        hasKeymap: page.hasKeymap,
      })),
    );
    return { images };
  });

  // Read a page's truth sidecar, or report that none exists yet.
  router.get('/api/labels', async (_params, request) => {
    const page = await resolveKeymapPage(dataDir, request.query.id);
    if (!page) throw new HTTPError(400, `no such key map: ${request.query.id}`);
    try {
      return JSON.parse(await readFile(page.labelsPath, 'utf8'));
    } catch {
      return { exists: false };
    }
  });

  // Write a page's truth sidecar, creating the volume's truth/ directory on
  // first save. The server fills in `image`, since it alone knows which file
  // the page's stem resolved to.
  router.put('/api/labels', async (_params, body, request) => {
    const page = await resolveKeymapPage(dataDir, request.query.id);
    if (!page) throw new HTTPError(400, `no such key map: ${request.query.id}`);
    const sidecar = {
      image: basename(page.imagePath),
      width: body.width,
      height: body.height,
      labels: body.labels,
    };
    await mkdir(dirname(page.labelsPath), { recursive: true });
    await writeFile(page.labelsPath, JSON.stringify(sidecar, null, 2) + '\n');
    return { ok: true };
  });
}
