/**
 * IIIF image serving + the volume/annotation JSON API.
 *
 * `registerIiifImages` mounts the raw binary endpoints (the express-iiif image
 * service under `/iiif`); `registerIiifApi` registers the typed JSON endpoints
 * (`/iiif-api/*`) on the shared crosswalk router.
 */

import { existsSync } from 'fs';
import { readdir, readFile, stat } from 'fs/promises';
import { createRequire } from 'module';
import { join } from 'path';
import type { Express } from 'express';
import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import {
  pageImageFile,
  legacyPageImageFile,
  type LocalPageImage,
  type PageImage,
  imageStemsByLowercase,
  rewriteAnnotationPage,
  serviceUrlToPageKey,
  type AnnotationFileInfo,
  type GeorefAnnotationPage,
  type VolumeInfo,
} from './iiifAnnotations.ts';
import { jpegDimensions } from './jpegDimensions.ts';
import { isS3Uri } from './s3Objects.ts';
import { s3Annotation } from './s3Routes.ts';
import { pngDimensions } from './pngDimensions.ts';
import {
  parseCompareFooter,
  parseCompareTxt,
  parseLandByPage,
  parseMissingTruthKeys,
} from './compareTxt.ts';
import {
  findVolumes,
  pageImageStems,
  volumePages,
  withRunPanels,
} from './adjacencyTruth.ts';
import { keymapAnnotation } from './keymapAnnotation.ts';
import { keymapInfos } from './keymapInfos.ts';
import {
  isRunDir,
  runArtifactDir,
  runArtifactStems,
  splitRunPath,
} from './runArtifacts.ts';
import { withTiles } from './iiifAnnotations.ts';
import { isSafeSegment, isSafeVolume } from './volumePaths.ts';

const require = createRequire(import.meta.url);
const iiif = require('express-iiif').default;

const PAGE_IMAGE_PATTERN = /^p\d+[a-z]?\.jpg$/i;

// A failed-georef sidecar name -> [full, stem, kind], e.g.
// "p1452.georef-nofit.json" -> ["…", "p1452", "nofit"].
const GEOREF_SIDECAR_PATTERN = /^(.+)\.georef(?:-[a-z0-9-]+)?\.json$/i;

// Read a *.iiif.json if it is a georeference AnnotationPage, else null.
/**
 * Item counts for annotation files, keyed by path and invalidated by mtime.
 *
 * The volume list parses every `*.iiif.json` under data/ purely to show an item
 * count beside each file: 633 files and 175 MB of JSON at the time of writing,
 * about half a second, on every request. Counts cannot change without the file
 * changing, so a stat is enough to reuse one (#288).
 */
const itemCountCache = new Map<
  string,
  { mtimeMs: number; itemCount: number; oimSlug?: string }
>();

/**
 * OIM map slug from a truth file's mosaic id, e.g.
 * ".../iiif/mosaic/sanborn09064_008/main-content/" -> "sanborn09064_008".
 *
 * Only the VOLUME is derivable offline. A per-page link would need OIM's
 * document id, and the annotation's own id is a IIIF resource id in a
 * different namespace: richmond p315 is resource 81359, and OIM's own page for
 * 81359 is a different sheet entirely (#298).
 */
function oimSlugOf(page: GeorefAnnotationPage): string | undefined {
  const id = (page as { id?: unknown }).id;
  const match =
    typeof id === 'string' ? id.match(/\/iiif\/mosaic\/([^/]+)\//) : null;
  return match?.[1];
}

async function annotationFacts(
  path: string,
  mtimeMs: number,
): Promise<{ itemCount: number; oimSlug?: string } | null> {
  const hit = itemCountCache.get(path);
  if (hit && hit.mtimeMs === mtimeMs) return hit;
  const page = await readAnnotationPage(path);
  if (!page) return null;
  const facts = {
    mtimeMs,
    itemCount: page.items.length,
    oimSlug: oimSlugOf(page),
  };
  itemCountCache.set(path, facts);
  return facts;
}

async function readAnnotationPage(
  path: string,
): Promise<GeorefAnnotationPage | null> {
  try {
    const data = JSON.parse(await readFile(path, 'utf8'));
    return data?.type === 'AnnotationPage' && Array.isArray(data.items)
      ? data
      : null;
  } catch {
    return null;
  }
}

// Page-image dimensions keyed by absolute path, invalidated by file mtime.
const dimensionsCache = new Map<
  string,
  { mtimeMs: number; dims: { width: number; height: number } }
>();

async function cachedJpegDimensions(
  path: string,
): Promise<{ width: number; height: number }> {
  const { mtimeMs } = await stat(path);
  const cached = dimensionsCache.get(path);
  if (cached && cached.mtimeMs === mtimeMs) return cached.dims;
  const dims = jpegDimensions(path);
  dimensionsCache.set(path, { mtimeMs, dims });
  return dims;
}

/**
 * Mount the raw IIIF image service (express-iiif) under `/iiif`.
 *
 * info.json responses are passed through {@link withTiles} first; see there
 * for why an advertised tileset is load-bearing for the map viewer.
 */
const PAGE_IMAGES: readonly PageImage[] = ['page', 'region', 'roadprob'];

// The annotation route's `image` query value, the sheet when absent; anything
// else is a client error rather than a silent fall-through to the sheet.
function pageImageOf(value: unknown): PageImage {
  const image = value ?? 'page';
  if (!PAGE_IMAGES.includes(image as PageImage)) {
    throw new HTTPError(400, `invalid image: ${String(value)}`);
  }
  return image as PageImage;
}

// A page's alternate image (its P(region) or P(road) map) at the file's own
// size when it is on disk, else null so the caller serves the sheet. P(road)
// maps are JPEG sidecars beside the page since #354, with the PNGs older runs
// left under artifacts/ as a fallback.
function alternatePageImage(
  volumeDir: string,
  image: Exclude<PageImage, 'page'>,
  imageKey: string,
): LocalPageImage | null {
  const candidates =
    image === 'roadprob'
      ? [pageImageFile(image, imageKey), legacyPageImageFile(imageKey)]
      : [pageImageFile(image, imageKey)];
  for (const file of candidates) {
    const path = join(volumeDir, file);
    try {
      const size = file.endsWith('.jpg')
        ? jpegDimensions(path)
        : pngDimensions(path);
      return { ...size, file };
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Annotation files a volume's mirror runs published, volume-relative:
 * `runs/corpus-v1/mapsnap.iiif.json`, `runs/corpus-v1/mapsnap.keymap.iiif.json`.
 *
 * A volume synced from the mirror (scripts/pull_item.py) keeps each run under
 * `runs/<tag>/`; one without that directory simply has none.
 */
async function runAnnotationFiles(volumeDir: string): Promise<string[]> {
  let runs: string[];
  try {
    runs = await readdir(join(volumeDir, 'runs'));
  } catch {
    return [];
  }
  const files = await Promise.all(
    runs.map(async (run) => {
      try {
        return (await readdir(join(volumeDir, 'runs', run)))
          .filter((file) => file.endsWith('.iiif.json'))
          .map((file) => `runs/${run}/${file}`);
      } catch {
        return [];
      }
    }),
  );
  return files.flat();
}

// The optional `run` of a volume-scoped query, checked; undefined when absent.
function runOf(run: unknown): string | undefined {
  if (run === undefined || run === '') return undefined;
  if (!isRunDir(run)) throw new HTTPError(400, `invalid run: ${String(run)}`);
  return run;
}

export function registerIiifImages(app: Express, dataDir: string): void {
  mountIiifImages(app, '/iiif', dataDir);
}

/**
 * A IIIF image server over `imageDir`, with tiles advertised on info.json.
 *
 * `before` runs first and may put the requested file in place; that is how the
 * S3 mount fetches a scan on demand rather than mirroring a whole item up
 * front. Any error it throws becomes a 502, since by then the only thing that
 * can have gone wrong is the fetch.
 */
export function mountIiifImages(
  app: Express,
  mount: string,
  imageDir: string,
  before?: (identifier: string) => Promise<void>,
): void {
  app.use(mount, (request, response, next) => {
    if (request.path.endsWith('/info.json')) {
      const json = response.json.bind(response);
      response.json = (body: unknown) =>
        json(
          body && typeof body === 'object'
            ? withTiles(body as Record<string, unknown>)
            : body,
        );
    }
    if (!before) return next();
    const identifier = iiifIdentifierOf(request.path);
    if (!identifier) return next();
    before(identifier).then(
      () => next(),
      (error: unknown) => {
        response.status(502).json({ error: String(error) });
      },
    );
  });
  app.use(mount, iiif({ imageDir }));
}

/**
 * The image a IIIF request names, or null if the path is not one.
 *
 * Two shapes reach the mount: `<identifier>/info.json`, and the image request
 * `<identifier>/{region}/{size}/{rotation}/{quality}.{format}`. The identifier
 * is whatever comes before those, and it is a path because these servers are
 * mounted over a directory tree.
 */
export function iiifIdentifierOf(path: string): string | null {
  const parts = path.split('/').filter(Boolean);
  if (parts[parts.length - 1] === 'info.json') {
    return parts.slice(0, -1).join('/') || null;
  }
  if (parts.length >= 5) return parts.slice(0, -4).join('/') || null;
  return null;
}

/** Register the typed volume/annotation JSON API (`/iiif-api/*`). */
export function registerIiifApi(
  router: TypedRouter<API>,
  dataDir: string,
  s3CacheDir: string,
): void {
  // Volume directories that have local page images and annotation files.
  //
  // findVolumes descends into a multi-volume atlas, so `brooklyn_1904-1908/vol13`
  // is listed under that name rather than being missed because its parent holds no
  // page images (#228). It stops at the first page-bearing directory on a branch,
  // so a volume's own `raw/` is never listed as a volume.
  router.get('/iiif-api/volumes', async () => {
    // Every volume, and every annotation within it, is read CONCURRENTLY.
    // Serially this walked ~18 volumes x several annotation files each,
    // parsing every one in full for its item count, and took long enough that
    // the volume picker arrived after the map had drawn (#288).
    const names = await findVolumes(dataDir);
    const built = await Promise.all(
      names.map(async (name): Promise<VolumeInfo | null> => {
        const files = await readdir(join(dataDir, name));
        const pageCount = files.filter((f) =>
          PAGE_IMAGE_PATTERN.test(f),
        ).length;
        if (pageCount === 0) return null;
        let oimSlug: string | undefined;
        const annotationFiles = [
          ...files.filter((f) => f.endsWith('.iiif.json')),
          ...(await runAnnotationFiles(join(dataDir, name))),
        ];
        const annotations = (
          await Promise.all(
            annotationFiles.map(
              async (file): Promise<AnnotationFileInfo | null> => {
                const path = join(dataDir, name, file);
                const info = await stat(path);
                const facts = await annotationFacts(path, info.mtimeMs);
                if (!facts) return null;
                if (file === 'main.iiif.json' && facts.oimSlug) {
                  oimSlug = facts.oimSlug;
                }
                return {
                  name: file,
                  modifiedMs: Math.round(info.mtimeMs),
                  itemCount: facts.itemCount,
                };
              },
            ),
          )
        ).filter((a): a is AnnotationFileInfo => a !== null);
        if (annotations.length === 0) return null;
        annotations.sort((a, b) => b.modifiedMs - a.modifiedMs);
        return { name, pageCount, annotations, oimSlug };
      }),
    );
    const volumes = built.filter((v): v is VolumeInfo => v !== null);
    volumes.sort((a, b) => a.name.localeCompare(b.name));
    return { volumes };
  });

  // Serve an AnnotationPage rewritten to target this server's /iiif endpoint.
  // The path is repo-root-relative like the app's ?files= param, so a leading
  // "data/" is tolerated (dataDir already points at the data directory).
  router.get('/iiif-api/annotation', async (_params, request) => {
    const rawPath = request.query.path;
    // An s3:// path is a run's own output read straight out of the mirror; the
    // rewrite is the same, only the pages come from the bucket.
    if (isS3Uri(rawPath)) {
      const origin = `${request.protocol}://${request.get('host')}`;
      return s3Annotation(rawPath, origin, s3CacheDir);
    }
    const image = pageImageOf(request.query.image);
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const annotationPath = join(dataDir, relativePath);
    const page = await readAnnotationPage(annotationPath);
    if (!page) {
      throw new HTTPError(
        404,
        `not found or not an AnnotationPage: ${rawPath}`,
      );
    }
    // A mirror run's annotation sits in `<volume>/runs/<tag>/`, but its pages
    // are the volume's own scans, at the volume root.
    const { volume: volumePath } = splitRunPath(relativePath);
    const volumeDir = join(dataDir, volumePath);
    const stems = await imageStemsByLowercase(volumeDir);
    const localPages = new Map<string, LocalPageImage>();
    const fallbacks: string[] = [];
    for (const item of page.items) {
      // Same (url, label) pair rewriteAnnotationPage will use: the label
      // carries the split-panel variant and, for volumes that link no image
      // service, the page number itself. Deriving the key differently here
      // keys localPages under names the rewrite never looks up, and every
      // page reports "missing-image".
      const derived = serviceUrlToPageKey(
        item?.target?.source?.id,
        String(item?.label ?? item?.id ?? ''),
        String(item?.id ?? ''),
      );
      // A split panel is georeferenced against its parent sheet, so the image
      // to measure and serve is the parent (see rewriteAnnotationPage).
      const parent = derived?.replace(/__\d+$/, '');
      if (!parent) continue;
      // Volumes disagree about the case of a lettered suffix -- Chicago's
      // 0103W is p103w.jpg on disk, Asheville's 0033A is p33A.jpg -- so the
      // derived key's case is a guess. Resolve it against the directory and
      // use the real stem, or every link the viewer builds from it 404s on a
      // case-sensitive filesystem (and reads the wrong name on a forgiving one).
      const imageKey = stems.get(parent.toLowerCase()) ?? parent;
      if (localPages.has(imageKey)) continue;
      try {
        const sheet = await cachedJpegDimensions(
          join(volumeDir, `${imageKey}.jpg`),
        );
        const alternate =
          image === 'page'
            ? null
            : alternatePageImage(volumeDir, image, imageKey);
        if (image !== 'page' && !alternate) fallbacks.push(imageKey);
        localPages.set(imageKey, alternate ?? sheet);
      } catch {
        // No local image for this page; rewriteAnnotationPage reports it.
      }
    }
    const serviceBaseUrl = `${request.protocol}://${request.get('host')}/iiif/${volumePath}`;
    const rewritten = rewriteAnnotationPage(
      page,
      localPages,
      serviceBaseUrl,
      stems,
    );
    return image === 'page'
      ? rewritten
      : { ...rewritten, imageFallbacks: fallbacks };
  });

  // Where a run's own per-page sidecars live, so the viewer can link to the files
  // that produced the annotation being looked at rather than to whatever the last
  // run happened to leave at the top level.
  router.get('/iiif-api/run-artifacts', async (_params, request) => {
    const rawPath = request.query.path;
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const artifactDir = runArtifactDir(relativePath);
    if (!artifactDir) return { dir: null, stems: [] };

    let files: string[];
    try {
      files = await readdir(join(dataDir, artifactDir));
    } catch {
      // A run with no saved sidecars is the common case, not an error.
      return { dir: null, stems: [] };
    }
    const stems = runArtifactStems(files);
    return stems.length > 0
      ? { dir: `data/${artifactDir}`, stems }
      : { dir: null, stems: [] };
  });

  // Per-page truth comparison from the annotation's `mapsnap compare` sidecar table
  // (`<name>.txt` next to `<name>.iiif.json`). Empty when there is no sidecar.
  router.get('/iiif-api/compare', async (_params, request) => {
    const rawPath = request.query.path;
    const relativePath = rawPath.replace(/^data\//, '');
    const parts = relativePath.split('/');
    if (
      !relativePath.endsWith('.iiif.json') ||
      parts.length < 2 ||
      !parts.every(isSafeSegment)
    ) {
      throw new HTTPError(400, `invalid path: ${rawPath}`);
    }
    const txtPath = join(
      dataDir,
      relativePath.replace(/\.iiif\.json$/, '.txt'),
    );
    try {
      const text = await readFile(txtPath, 'utf8');
      return {
        pages: parseCompareTxt(text),
        missing: parseMissingTruthKeys(text),
        ...((land) => (land ? { landKm2ByPage: land } : {}))(
          parseLandByPage(text),
        ),
        footer: parseCompareFooter(text),
      };
    } catch {
      return { pages: [], missing: [], footer: '' };
    }
  });

  // A volume's adjacency.json (per-page sheet-number claims + the mutual-edge graph),
  // for the viewer's adjacency overlay. Null when the volume has no adjacency data.
  // With a `run`, that run's own adjacency.json comes first: a mirror run
  // writes one under runs/<tag>/ and the volume root has none.
  router.get('/iiif-api/adjacency', async (_params, request) => {
    const { volume } = request.query;
    const run = runOf(request.query.run);
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    const candidates = [
      ...(run ? [join(dataDir, volume, run, 'adjacency.json')] : []),
      join(dataDir, volume, 'adjacency.json'),
    ];
    for (const path of candidates) {
      try {
        return { adjacency: JSON.parse(await readFile(path, 'utf8')) };
      } catch {
        continue;
      }
    }
    return { adjacency: null };
  });

  // The boundary of the OSM relation this volume's streets were downloaded from
  // (data/<volume>/r<id>.json, saved by the coverage sweep). The viewer draws it
  // so a page whose ground falls OUTSIDE the download is visible as such: those
  // pages' streets are missing from the vocabulary entirely, which is why
  // richmond p383 and fargo's Moorhead sheets cannot be fit at all.
  router.get('/iiif-api/osm-relation', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    try {
      const dir = join(dataDir, volume);
      // Which relation the streets came from is recorded in the volume's own
      // manifest, so a leftover r<id>.json from an earlier download is ignored
      // rather than drawing a boundary the current streets did not come from.
      const manifest = JSON.parse(
        await readFile(join(dir, 'mapsnap.json'), 'utf8'),
      );
      const name: unknown = manifest?.params?.relation;
      if (typeof name !== 'string' || !/^r\d+$/.test(name)) {
        return { relation: null };
      }
      const doc = JSON.parse(await readFile(join(dir, `${name}.json`), 'utf8'));
      const element = doc.elements?.[0];
      if (!element) return { relation: null };
      const ways = (element.members ?? [])
        .filter((m: any) => m.type === 'way' && Array.isArray(m.geometry))
        .map((m: any) =>
          m.geometry.map((p: any) => [p.lon, p.lat] as [number, number]),
        )
        .filter((w: unknown[]) => w.length >= 2);
      // download-osm records the buffer it used on the saved boundary, so the
      // ring can say whether it traces the administrative line or the (larger)
      // area actually downloaded. Absent on volumes fetched before buffering.
      const bufferRaw = element.tags?.['mapsnap:buffer_m'];
      const bufferM = bufferRaw == null ? null : Number(bufferRaw);
      return {
        relation: {
          id: name,
          bufferM: Number.isFinite(bufferM) ? bufferM : null,
          name: element.tags?.name ?? null,
          ways,
        },
      };
    } catch {
      return { relation: null };
    }
  });

  // A volume's page files: every page-image stem, plus every georef sidecar each
  // page has, so the viewer can link to all of them and — for a volume with no
  // truth annotation — work out which pages went unplaced. ?volume=<dir> →
  // { pages: ["p1", …], georefs: { "p12": ["p12.georef.json", "p12.georef-snap.json"] } }.
  //
  // With a `run`, the georef sidecars are that run's (`runs/<tag>/`), and the
  // split panels its sidecars name count as pages: a volume synced from the
  // mirror has no panel images to find them by.
  router.get('/iiif-api/failed-georefs', async (_params, request) => {
    const { volume } = request.query;
    const run = runOf(request.query.run);
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    let files: string[];
    try {
      files = await readdir(join(dataDir, volume, run ?? ''));
    } catch {
      throw new HTTPError(404, `no such volume: ${volume}`);
    }
    const georefs: Record<string, string[]> = {};
    for (const file of files) {
      const match = file.match(GEOREF_SIDECAR_PATTERN);
      if (match && match[1]) (georefs[match[1]] ??= []).push(file);
    }
    // Plain `georef.json` first, then the variants alphabetically, so the
    // RANSAC fit heads the list and the channels follow in a stable order.
    for (const list of Object.values(georefs)) {
      list.sort((a, b) => a.length - b.length || a.localeCompare(b));
    }
    // volumePages drops a split sheet in favour of its panels, so a sheet whose panels
    // all fitted is not reported as an unplaced page.
    const pages = await volumePages(dataDir, volume);
    // Page images are always at the volume root, whichever directory the
    // sidecars were listed from.
    const rootFiles = run
      ? await readdir(join(dataDir, volume)).catch(() => [])
      : files;
    return {
      georefs,
      pages: run ? withRunPanels(pages, files) : pages,
      images: pageImageStems(rootFiles),
    };
  });

  // A volume's key-map sheets and which visualization sidecars each has, so the viewer can link
  // to them and draw the key-map underlay (see keymapInfos). ?volume=<dir> → { keymaps: [...] }.
  router.get('/iiif-api/keymaps', async (_params, request) => {
    const { volume } = request.query;
    if (!isSafeVolume(volume)) {
      throw new HTTPError(400, `invalid volume: ${volume}`);
    }
    const serviceBaseUrl = `${request.protocol}://${request.get('host')}/iiif/${volume}/raw`;
    return {
      keymaps: await keymapInfos(join(dataDir, volume, 'raw'), serviceBaseUrl),
    };
  });

  // One key map as a georeference annotation, for the underlay: the sheet or
  // its P(road) map, warped by a thin-plate spline through the sheet's own
  // GCPs -- the model keymap-snap places pages in (see keymapAnnotation).
  router.get('/iiif-api/keymap-annotation', async (_params, request) => {
    const { volume, stem, image } = request.query;
    if (!isSafeVolume(volume) || !isSafeSegment(stem)) {
      throw new HTTPError(400, `invalid key map: ${volume}/${stem}`);
    }
    if (image !== 'sheet' && image !== 'roadprob') {
      throw new HTTPError(400, `invalid image: ${image}`);
    }
    const rawDir = join(dataDir, volume, 'raw');
    let georef: unknown;
    try {
      georef = JSON.parse(
        await readFile(join(rawDir, `${stem}.georef.json`), 'utf8'),
      );
    } catch {
      throw new HTTPError(404, `no georef for key map ${volume}/${stem}`);
    }
    // The P(road) map is rendered in the sheet's own pixel frame, so one set
    // of GCPs places either image.
    const candidates =
      image === 'roadprob'
        ? [`${stem}.roadprob.png`]
        : [`${stem}.jpg`, `${stem}.png`];
    const file = candidates.find((name) => existsSync(join(rawDir, name)));
    if (!file) {
      throw new HTTPError(
        404,
        `no ${image} image for key map ${volume}/${stem}`,
      );
    }
    const serviceUrl = `${request.protocol}://${request.get('host')}/iiif/${volume}/raw/${file}`;
    const page = keymapAnnotation(
      georef,
      serviceUrl,
      `keymap:${volume}/${stem}/${image}`,
    );
    if (!page) {
      throw new HTTPError(
        404,
        `key map ${volume}/${stem} has no usable georeference`,
      );
    }
    return page;
  });
}
