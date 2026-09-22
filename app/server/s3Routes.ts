/**
 * Viewing a run's annotation straight out of the mirror.
 *
 * `?view=iiif&iiif=s3://mapsnap-sanborn/…/runs/corpus-v1/mapsnap.iiif.json`
 * renders the ordinary volume viewer against an annotation the corpus run
 * published, with its scans served from the bucket instead of loc.gov. The
 * annotations point their image services at loc.gov, which is unreliable
 * enough that a 120-page volume mostly draws holes.
 *
 * The rewrite is the same one the local viewer uses -- measure each page, point
 * its service at a server we control, rescale the GCPs into that image's
 * frame. Only where the pages come from differs, so a run's output is debugged
 * through exactly the machinery that debugs a local volume.
 *
 * Scans are cached on first request rather than mirrored up front: a volume is
 * ~100 MB at 25% scale and the viewer only ever draws the pages in view.
 */

import { mkdir, open } from 'fs/promises';
import { join } from 'path';
import type { Express } from 'express';
import { HTTPError } from 'crosswalk';

import {
  rewriteAnnotationPage,
  serviceUrlToPageKey,
  type GeorefAnnotationPage,
  type LocalPageImage,
  type RewriteResult,
} from './iiifAnnotations.ts';
import { jpegDimensionsFromBuffer } from './jpegDimensions.ts';
import { mountIiifImages } from './iiifRoutes.ts';
import {
  cachePathOf,
  ensureCached,
  imagePrefixOf,
  parseS3Uri,
  readCachedS3Text,
  readS3Head,
  uriFromCacheRelative,
} from './s3Objects.ts';

/** Bytes that reach a JPEG's start-of-frame marker. */
const HEAD_BYTES = 128 * 1024;

/** One sheet of an item's `metadata.json`, as the mirror writes it. */
interface MirrorSheet {
  key?: string;
  width?: number;
  height?: number;
}

/**
 * Page sizes from the item's own `metadata.json`, or an empty map.
 *
 * The mirror recorded every sheet's scaled width and height when it wrote the
 * scan, so one object answers the whole volume. Measuring instead costs a
 * ranged read per page -- 120 of them before the viewer can draw anything --
 * and produces the same numbers.
 */
async function mirrorPageSizes(
  cacheRoot: string,
  bucket: string,
  prefix: string,
): Promise<Map<string, LocalPageImage>> {
  const sizes = new Map<string, LocalPageImage>();
  let sheets: MirrorSheet[];
  try {
    const text = await readCachedS3Text(cacheRoot, {
      bucket,
      key: `${prefix}/metadata.json`,
    });
    sheets = (JSON.parse(text) as { sheets?: MirrorSheet[] }).sheets ?? [];
  } catch {
    // An item mirrored before metadata.json carried sizes, or not mirrored at
    // all: every page falls through to being measured.
    return sizes;
  }
  for (const sheet of sheets) {
    if (sheet.key && sheet.width && sheet.height) {
      sizes.set(sheet.key, { width: sheet.width, height: sheet.height });
    }
  }
  return sizes;
}

/** A JPEG's dimensions from its first bytes, on disk or in the bucket. */
async function measurePage(
  cacheRoot: string,
  uri: { bucket: string; key: string },
): Promise<LocalPageImage> {
  const cached = cachePathOf(cacheRoot, uri);
  try {
    // Already fetched: read the head off the disk rather than the network.
    const handle = await open(cached, 'r');
    try {
      const buffer = Buffer.alloc(HEAD_BYTES);
      const { bytesRead } = await handle.read(buffer, 0, HEAD_BYTES, 0);
      return jpegDimensionsFromBuffer(buffer.subarray(0, bytesRead));
    } finally {
      await handle.close();
    }
  } catch {
    return jpegDimensionsFromBuffer(await readS3Head(uri, HEAD_BYTES));
  }
}

/**
 * Serve a run's JSON sidecars out of the mirror, cached like everything else.
 *
 * The viewer needs a split sheet's `pN.panels.json` to grey the panels a
 * selected one was cut from. For a local volume it reads that from
 * `/data/<volume>/`; an annotation in the bucket has no such directory, and
 * without the panels the viewer falls back to punching the panel's own ring
 * out of the sheet rectangle -- a hole whose edges lie on the outer ring,
 * which degenerates MapLibre's triangulation into visible wedges.
 */
export function registerS3Objects(app: Express, cacheRoot: string): void {
  app.get('/s3-api/object', (request, response) => {
    void (async () => {
      const uri = String(request.query.uri ?? '');
      const object = parseS3Uri(uri);
      if (!object || !object.key.endsWith('.json')) {
        response.status(400).json({ error: `not a JSON object: ${uri}` });
        return;
      }
      try {
        response
          .type('application/json')
          .send(await readCachedS3Text(cacheRoot, object));
      } catch (error) {
        response.status(404).json({ error: String(error) });
      }
    })();
  });
}

/** Mount the on-demand image server for cached mirror scans. */
export function registerS3IiifImages(app: Express, cacheRoot: string): void {
  mountIiifImages(app, '/s3-iiif', cacheRoot, async (identifier) => {
    const uri = uriFromCacheRelative(identifier);
    if (!uri) throw new Error(`not an object path: ${identifier}`);
    await ensureCached(cacheRoot, uri);
  });
}

/**
 * The rewritten annotation for an object in the mirror.
 *
 * Page sizes come from the item's `metadata.json`, which the mirror wrote with
 * every sheet's scaled dimensions -- one object for the volume rather than a
 * ranged read per page. Anything it does not name is measured: from the cached
 * scan when there is one, and only otherwise from the bucket. A page whose scan
 * is missing entirely is left to the rewrite to report, exactly as a missing
 * local file is.
 */
export async function s3Annotation(
  uri: string,
  serviceOrigin: string,
  cacheRoot: string,
): Promise<RewriteResult> {
  const object = parseS3Uri(uri);
  if (!object || !object.key.endsWith('.iiif.json')) {
    throw new HTTPError(400, `not an annotation object: ${uri}`);
  }
  let page: GeorefAnnotationPage;
  try {
    page = JSON.parse(
      await readCachedS3Text(cacheRoot, object),
    ) as GeorefAnnotationPage;
  } catch (error) {
    throw new HTTPError(404, `could not read ${uri}: ${String(error)}`);
  }
  if (!Array.isArray(page?.items)) {
    throw new HTTPError(404, `not an AnnotationPage: ${uri}`);
  }

  const prefix = imagePrefixOf(object.key);
  const pages = new Map<string, LocalPageImage>();
  const wanted = new Set<string>();
  for (const item of page.items) {
    const derived = serviceUrlToPageKey(
      item?.target?.source?.id,
      String(item?.label ?? item?.id ?? ''),
      String(item?.id ?? ''),
    );
    // A split panel is georeferenced against its parent sheet, so the scan to
    // measure and serve is the parent's -- the same rule the local viewer
    // follows, and for the same reason: the GCPs are in parent pixels.
    const parent = derived?.replace(/__\d+$/, '');
    if (parent) wanted.add(parent);
  }
  const recorded = await mirrorPageSizes(cacheRoot, object.bucket, prefix);
  await Promise.all(
    [...wanted].map(async (key) => {
      const known = recorded.get(key);
      if (known) {
        pages.set(key, known);
        return;
      }
      try {
        pages.set(
          key,
          await measurePage(cacheRoot, {
            bucket: object.bucket,
            key: `${prefix}/${key}.jpg`,
          }),
        );
      } catch {
        // No scan for this page in the mirror; the rewrite reports it as
        // missing-image, which is what the viewer already knows how to show.
      }
    }),
  );

  await mkdir(join(cacheRoot, object.bucket, prefix), { recursive: true });
  const serviceBaseUrl = `${serviceOrigin}/s3-iiif/${object.bucket}/${prefix}`;
  return rewriteAnnotationPage(page, pages, serviceBaseUrl);
}
