/**
 * Reading mirror objects out of S3 for the debugger.
 *
 * The annotations the corpus run publishes point their image services at
 * loc.gov, which is too unreliable to debug against: a volume viewer loading
 * 120 pages from it mostly shows holes. The mirror holds the same scans, so
 * the viewer can be pointed at `s3://…/mapsnap.iiif.json` instead and the
 * images served from the bucket.
 *
 * Objects are fetched through the `aws` CLI rather than an SDK. The debug
 * server only ever runs on a machine that already has a configured profile,
 * and shelling out inherits it -- credentials, SSO, region and all -- without
 * adding a dependency or a second place for them to be wrong.
 */

import { execFile } from 'child_process';
import { mkdir, readFile, rename, stat } from 'fs/promises';
import { dirname, join } from 'path';
import { promisify } from 'util';

const run = promisify(execFile);

/** An `s3://bucket/key` reference, split. */
export interface S3Uri {
  bucket: string;
  key: string;
}

/** Parse `s3://bucket/key/parts`, or null if it is not one. */
export function parseS3Uri(uri: string): S3Uri | null {
  const match = /^s3:\/\/([^/]+)\/(.+)$/.exec(uri.trim());
  if (!match) return null;
  const [, bucket, key] = match;
  if (!bucket || !key || key.includes('..')) return null;
  return { bucket, key };
}

/** Whether a viewer path names an S3 object rather than a file under data/. */
export function isS3Uri(path: string): boolean {
  return path.trim().startsWith('s3://');
}

/**
 * The item directory holding a run's page images, given the run's annotation.
 *
 * A run publishes to `<item>/runs/<tag>/mapsnap.iiif.json`, but the scans it
 * describes stay at the item root -- `loc-fit` uploads sidecars per run and
 * images never. So the annotation's own directory is the wrong place to look
 * for `p1.jpg`, and the run segments have to come off.
 */
export function imagePrefixOf(key: string): string {
  const parts = key.split('/').filter(Boolean);
  parts.pop(); // the annotation file itself
  if (parts.length >= 2 && parts[parts.length - 2] === 'runs') {
    parts.splice(-2, 2);
  }
  return parts.join('/');
}

/**
 * The cache path an S3 object is materialised at, under `root`.
 *
 * Bucket first, so two buckets with the same key never collide, and so the
 * path a IIIF request carries round-trips back to the object it came from.
 */
export function cachePathOf(root: string, uri: S3Uri): string {
  return join(root, uri.bucket, uri.key);
}

/** The object a `/s3-iiif/<bucket>/<key>` request names. */
export function uriFromCacheRelative(relative: string): S3Uri | null {
  const parts = relative.split('/').filter(Boolean);
  if (parts.length < 2 || parts.some((part) => part === '..')) return null;
  const [bucket, ...rest] = parts;
  return bucket ? { bucket, key: rest.join('/') } : null;
}

/** Credentials the CLI could not use, as opposed to a missing object. */
function isCredentialFailure(message: string): boolean {
  return /ExpiredToken|InvalidGrant|authorization grant is invalid|expired|Unable to locate credentials|AccessDenied|InvalidClientTokenId/i.test(
    message,
  );
}

async function aws(args: string[], encoding: 'utf8' | 'buffer' = 'utf8') {
  const profile = process.env.MAPSNAP_S3_PROFILE;
  const full = profile ? [...args, '--profile', profile] : args;
  try {
    return await run('aws', full, {
      encoding: encoding === 'buffer' ? 'buffer' : 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    } as never);
  } catch (error) {
    const message = String(
      (error as { stderr?: string })?.stderr ||
        (error as Error)?.message ||
        error,
    );
    // The default profile here is a login session that lapses every ten or
    // twenty minutes, so this is the failure to expect, and "Command failed"
    // buried under a stack trace is not much of a clue.
    if (isCredentialFailure(message)) {
      throw new Error(
        `the aws CLI could not read this object: ${message.trim().split('\n').pop()}\n` +
          'Start the debug server with a working profile, e.g. ' +
          'AWS_PROFILE=mapsnap npm run server, or set MAPSNAP_S3_PROFILE.',
      );
    }
    throw error;
  }
}

/**
 * The first `bytes` of an object.
 *
 * Enough to read a JPEG's dimensions out of its start-of-frame marker, which
 * is what the annotation rewrite needs. Pulling whole scans to measure them
 * would mean ~100 MB before a 120-page volume drew anything.
 */
export async function readS3Head(uri: S3Uri, bytes: number): Promise<Buffer> {
  const target = join(
    process.env.TMPDIR ?? '/tmp',
    `mapsnap-head-${process.pid}-${Math.random().toString(36).slice(2)}`,
  );
  await aws([
    's3api',
    'get-object',
    '--bucket',
    uri.bucket,
    '--key',
    uri.key,
    '--range',
    `bytes=0-${bytes - 1}`,
    target,
  ]);
  const { readFile, rm } = await import('fs/promises');
  try {
    return await readFile(target);
  } finally {
    await rm(target, { force: true });
  }
}

/**
 * Ensure an object is in the cache, and return where.
 *
 * Downloads to a temporary name and renames into place, so a request that
 * arrives while another is still fetching never reads a half-written file --
 * a IIIF viewer asks for many tiles of the same page at once.
 */
export async function ensureCached(root: string, uri: S3Uri): Promise<string> {
  const destination = cachePathOf(root, uri);
  try {
    const existing = await stat(destination);
    if (existing.size > 0) return destination;
  } catch {
    // Not cached yet.
  }
  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.${process.pid}.${Math.random().toString(36).slice(2)}`;
  await aws(['s3', 'cp', `s3://${uri.bucket}/${uri.key}`, temporary]);
  await rename(temporary, destination);
  return destination;
}
/**
 * An object's contents as text, cached on disk like the scans are.
 *
 * Both objects the viewer reads per volume -- a run's AnnotationPage and the
 * item's metadata.json -- are written once and not rewritten afterwards, so
 * re-fetching them on every page load buys nothing but latency. A run
 * republished under the same tag is the one case that goes stale; delete the
 * cache directory (MAPSNAP_S3_CACHE, ~/.cache/mapsnap/s3 by default) to
 * refetch.
 */
export async function readCachedS3Text(
  root: string,
  uri: S3Uri,
): Promise<string> {
  return readFile(await ensureCached(root, uri), 'utf8');
}
