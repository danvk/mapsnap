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

import { mkdir } from 'fs/promises';
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
  ensureCached,
  imagePrefixOf,
  parseS3Uri,
  readS3Head,
  readS3Text,
  uriFromCacheRelative,
} from './s3Objects.ts';

/** Bytes pulled to read a JPEG's start-of-frame marker. */
const HEAD_BYTES = 128 * 1024;

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
 * Page sizes come from a ranged read of each scan's first 128 KB, which is
 * enough for the dimensions and avoids pulling a whole volume to draw its
 * first tile. A page whose scan is missing is left to the rewrite to report,
 * exactly as a missing local file is.
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
    page = JSON.parse(await readS3Text(object)) as GeorefAnnotationPage;
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
  await Promise.all(
    [...wanted].map(async (key) => {
      try {
        const head = await readS3Head(
          { bucket: object.bucket, key: `${prefix}/${key}.jpg` },
          HEAD_BYTES,
        );
        pages.set(key, jpegDimensionsFromBuffer(head));
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
