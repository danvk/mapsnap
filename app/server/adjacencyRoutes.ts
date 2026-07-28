/**
 * Typed JSON API for the adjacency-truth labeler (`/api/adjacency-*`).
 *
 * Page images are not served here: the labeler loads them through the static
 * `data/` route (the Vite serveDataDir plugin in development, express.static
 * in production), which both resolve `data/<volume>/<stem>.jpg` relative to
 * the app base.
 */

import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import {
  findVolumes,
  isSafePage,
  isSafeVolume,
  readTruth,
  volumePages,
  writePageTruth,
} from './adjacencyTruth.ts';
import type { ImageInfo } from '../src/keymap/types.ts';

/** Register the adjacency-truth endpoints on the shared crosswalk router. */
export function registerAdjacencyTruthApi(
  router: TypedRouter<API>,
  dataDir: string,
): void {
  // Volumes that have labelable pages, with their labelling progress.
  router.get('/api/adjacency-volumes', async () => {
    const volumes = [];
    for (const name of await findVolumes(dataDir)) {
      const pages = await volumePages(dataDir, name);
      if (!pages.length) continue;
      const truth = await readTruth(dataDir, name);
      const labeled = pages.filter((p) => truth[p]?.labels?.length).length;
      volumes.push({ name, pageCount: pages.length, labeledPages: labeled });
    }
    return { volumes };
  });

  // One volume's pages, each with its current label count.
  router.get('/api/adjacency-pages', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    const truth = await readTruth(dataDir, volume);
    const pages: ImageInfo[] = (await volumePages(dataDir, volume)).map(
      (stem) => {
        const labels = truth[stem]?.labels ?? [];
        const withText = labels.filter((l) => l.text.trim()).length;
        return { name: stem, withText, withoutText: labels.length - withText };
      },
    );
    return { pages };
  });

  // Read one page's labels, or report that none exist yet.
  router.get('/api/adjacency-labels', async (_params, request) => {
    const { volume, page } = request.query;
    if (!isSafeVolume(volume) || !isSafePage(page)) {
      throw new HTTPError(400, `invalid target: ${volume}/${page}`);
    }
    const entry = (await readTruth(dataDir, volume))[page];
    return entry ?? { exists: false };
  });

  // Write one page's labels into the volume's adjacency-truth.json.
  router.put('/api/adjacency-labels', async (_params, body, request) => {
    const { volume, page } = request.query;
    if (!isSafeVolume(volume) || !isSafePage(page)) {
      throw new HTTPError(400, `invalid target: ${volume}/${page}`);
    }
    await writePageTruth(dataDir, volume, page, {
      width: body.width,
      height: body.height,
      labels: body.labels,
    });
    return { ok: true };
  });
}
