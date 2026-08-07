/**
 * Rewriting of IIIF Georeference AnnotationPages to target local images.
 *
 * The pipeline's *.iiif.json files point each page at a loc.gov IIIF image
 * service, with GCP resourceCoords and the SvgSelector clipping polygon in
 * that full-resolution image's pixel space. These helpers rewrite an
 * AnnotationPage to point at a local IIIF Image API server instead, rescaling
 * all pixel coordinates into the local (downscaled) image's frame.
 *
 * Everything here is pure (no filesystem access); iiif-server.mjs wires it up.
 */

/** Dimensions of a local page image, keyed by page key in rewriteAnnotationPage. */
export interface LocalPageImage {
  width: number;
  height: number;
}

export interface GeorefSource {
  id: string;
  type: string;
  width: number;
  height: number;
}

export interface GeorefTarget {
  source: GeorefSource;
  selector?: { type: string; value: string };
  [key: string]: unknown;
}

export interface GcpFeature {
  type: string;
  properties: { resourceCoords?: number[]; [key: string]: unknown };
  geometry: unknown;
}

export interface GeorefAnnotationItem {
  id?: string;
  type: string;
  label?: string;
  metadata?: { label: string; value: string }[];
  target?: GeorefTarget;
  body?: { features?: GcpFeature[]; [key: string]: unknown };
  [key: string]: unknown;
}

export interface GeorefAnnotationPage {
  id?: string;
  type: string;
  label?: string;
  items: GeorefAnnotationItem[];
  [key: string]: unknown;
}

/** An annotation item that was dropped during rewriting, and why. */
export interface SkippedItem {
  label: string;
  pageKey: string | null;
  /** 'not-a-page': cover/index/no source URL; 'missing-image': no local jpg. */
  reason: 'not-a-page' | 'missing-image';
}

export interface RewriteResult {
  annotation: GeorefAnnotationPage;
  skipped: SkippedItem[];
}

/** One *.iiif.json AnnotationPage file available in a volume directory. */
export interface AnnotationFileInfo {
  name: string;
  modifiedMs: number;
  itemCount: number;
}

/** A volume directory with local page images and georeference annotations. */
export interface VolumeInfo {
  name: string;
  pageCount: number;
  annotations: AnnotationFileInfo[];
}

/** Response shape of GET /iiif-api/volumes. */
export interface VolumeListResponse {
  volumes: VolumeInfo[];
}

/** Response shape of GET /iiif-api/annotation?path=... */
export type RewrittenAnnotationResponse = RewriteResult;

/**
 * Extract the page key from an OIM annotation label, or null if it has none.
 *
 * Port of label_to_page_key in mapsnap/utils.py: the page identifier in the
 * last pipe-separated segment, lowercased, with a bracketed split variant
 * collapsed to a double underscore.
 *   "New Orleans, La. | 1951 | Vol. 5 p428 [2]"  → "p428__2"
 *   "Grand Rapids, Mich. | 1953 | Vol. 7 p844"   → "p844"
 * Un-numbered index sheets carry letter-only ids ("... pa [2]" → "pa__2").
 */
export function labelToPageKey(label: string): string | null {
  const lastPart = (label.split('|').pop() ?? '').trim();
  const match = lastPart.match(/\b(p\d+[a-z]?|p[a-z]{1,2})(?:\s*\[(\d+)\])?$/i);
  if (!match) return null;
  const page = (match[1] ?? '').toLowerCase();
  return match[2] ? `${page}__${match[2]}` : page;
}

/**
 * Split-panel number for an annotation item, or null for a whole page.
 *
 * The ID is authoritative and the label is the fallback, because the two
 * disagree on generated annotations: `mapsnap iiif` copies the label from the
 * truth item it matched, so all three of fargo p45's panels are labelled
 * "p45 [2]" while their ids correctly read `-0045__1/georef`, `__2`, `__3`.
 * Trusting the label there points every panel at the same (wrong) image.
 * A truth item has no `__N` in its id and carries the panel number only in its
 * label; a generated WHOLE page ends in `/georef` with no `__N`, and its label
 * may still carry a stray `[N]` copied from truth, which must be ignored.
 */
export function splitIndexFor(
  id: string | undefined,
  label: string | undefined,
): number | null {
  const idMatch = id?.match(/__(\d+)\//);
  if (idMatch) return Number(idMatch[1]);
  if (id?.includes('/georef')) return null; // generated whole page
  const labelMatch = label?.match(/\[(\d+)\]\s*$/);
  return labelMatch ? Number(labelMatch[1]) : null;
}

/**
 * Extract the page key for an annotation, or null for non-page images.
 *
 * Port of source_id_to_page_key in mapsnap/utils.py, and it needs BOTH the
 * service URL and the annotation label:
 *
 * - The URL carries the page number:
 *     "...:01790_01N_1950-0006N/info.json" → "p6n"
 *     "...:05791_02_1939-0027s"            → "p27s"
 *   Sanborn sb-format (5 digits then a suffix char, '0' meaning none):
 *     "...:sb001250" → "p125";  "...:sb00154s" → "p154s"
 * - The split-panel variant comes from {@link splitIndexFor}, which prefers the
 *   item id and falls back to the label; no URL records it. A truth
 *   annotation for one panel of a split sheet is labelled "... p844 [3]" and
 *   must resolve to "p844__3"; deriving "p844" instead silently collapses
 *   every panel of a sheet onto its parent, so several annotations claim one
 *   key and all but one are lost (228 panels across the 15 truth volumes).
 * - Some OIM volumes link no image service at all and carry `source.id: null`
 *   (Grand Rapids 1953 vol 7: 73 of 83 annotations). The label alone then
 *   resolves the page.
 *
 * Covers and indexes ("...-covr", "...-titl") return null when the label has
 * no page identifier either.
 */
export function serviceUrlToPageKey(
  url: string | null | undefined,
  label = '',
  id?: string,
): string | null {
  const splitIndex = splitIndexFor(id, label);
  const splitSuffix = splitIndex != null ? `__${splitIndex}` : '';
  if (!url) {
    const fromLabel = labelToPageKey(label);
    if (!fromLabel) return null;
    // labelToPageKey already appends the label's own variant; when the id
    // overrides it (or says there is none), rebuild from the parent key.
    return fromLabel.replace(/__\d+$/, '') + splitSuffix;
  }
  url = url.replace(/\/info\.json$/, '');
  const sbMatch = url.match(/:sb(\d{5})([a-z0-9])$/i);
  if (sbMatch) {
    const pageNum = parseInt(sbMatch[1] ?? '', 10);
    const suffixChar = (sbMatch[2] ?? '').toLowerCase();
    const suffix = suffixChar === '0' ? '' : suffixChar;
    return `p${pageNum}${suffix}${splitSuffix}`;
  }
  const match = url.match(/-0*(\d+)([a-z]*)$/i);
  if (!match) return null;
  return `p${match[1]}${(match[2] ?? '').toLowerCase()}${splitSuffix}`;
}

// Round to 1 decimal, rendering integral values without a trailing ".0".
function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

// Clamp value into [0, max], mapping -0 to 0.
function clamp(value: number, max: number): number {
  return Math.min(Math.max(value, 0), max);
}

/**
 * Rescale an SvgSelector's polygon points by (scaleX, scaleY).
 *
 * Points are clamped into [0, width] × [0, height]; the original files
 * occasionally contain values like "-0.0" or slightly out-of-bounds floats.
 */
export function rescaleSvgSelector(
  svg: string,
  scale: { scaleX: number; scaleY: number },
  bounds: { width: number; height: number },
): string {
  return svg.replace(/points="([^"]*)"/, (unused, points: string) => {
    const rescaled = points
      .trim()
      .split(/\s+/)
      .map((pair) => {
        const [x = 0, y = 0] = pair.split(',').map(Number);
        const newX = round1(clamp(x * scale.scaleX, bounds.width));
        const newY = round1(clamp(y * scale.scaleY, bounds.height));
        return `${newX},${newY}`;
      })
      .join(' ');
    return `points="${rescaled}"`;
  });
}

/**
 * Rewrite a loc.gov-targeting AnnotationPage to target a local IIIF service.
 *
 * localPages maps page keys (e.g. "p11") to local image dimensions;
 * serviceBaseUrl is the IIIF prefix for the volume, e.g.
 * "http://localhost:8182/iiif/brooklyn_ny_1906_vol_6". Each kept item gets
 * target.source pointed at `${serviceBaseUrl}/${pageKey}.jpg` with the local
 * dimensions, its resourceCoords and clipping polygon rescaled to match, and a
 * `page` metadata entry recording the page key. Items without a page key or a
 * local image are dropped and reported in `skipped`. The input is not mutated.
 */
export function rewriteAnnotationPage(
  page: GeorefAnnotationPage,
  localPages: Map<string, LocalPageImage>,
  serviceBaseUrl: string,
  /**
   * Real page stems by lowercased form, from {@link imageStemsByLowercase}.
   *
   * The case of a lettered suffix cannot be derived from the annotation --
   * Chicago's `...-0103W` is `p103w.jpg` on disk, Asheville's `...-0033A` is
   * `p33A.jpg` -- so the derived key is resolved against the volume's actual
   * filenames. Resolving here rather than in the caller keeps the keys this
   * function emits and the keys it looks up in `localPages` the same; splitting
   * that across two places silently turns every lettered page into
   * "missing-image".
   */
  stemsByLowercase?: Map<string, string>,
): RewriteResult {
  const result: GeorefAnnotationPage = structuredClone(page);
  const kept: GeorefAnnotationItem[] = [];
  const skipped: SkippedItem[] = [];
  for (const item of result.items ?? []) {
    const label = String(item.label ?? item.id ?? '');
    const target = item.target;
    const source = target?.source;
    const derived = serviceUrlToPageKey(
      source?.id,
      label,
      String(item.id ?? ''),
    );
    const pageKey = derived
      ? (stemsByLowercase?.get(derived.toLowerCase()) ?? derived)
      : derived;
    if (!pageKey || !target || !source?.width || !source.height) {
      skipped.push({ label, pageKey, reason: 'not-a-page' });
      continue;
    }
    const local = localPages.get(pageKey);
    if (!local) {
      skipped.push({ label, pageKey, reason: 'missing-image' });
      continue;
    }
    const scaleX = local.width / source.width;
    const scaleY = local.height / source.height;
    target.source = {
      id: `${serviceBaseUrl}/${pageKey}.jpg`,
      type: 'ImageService3',
      width: local.width,
      height: local.height,
    };
    for (const feature of item.body?.features ?? []) {
      const coords = feature.properties?.resourceCoords;
      if (coords && coords.length >= 2) {
        feature.properties.resourceCoords = [
          round1((coords[0] ?? 0) * scaleX),
          round1((coords[1] ?? 0) * scaleY),
        ];
      }
    }
    const selector = target.selector;
    if (selector?.type === 'SvgSelector') {
      selector.value = rescaleSvgSelector(
        selector.value,
        { scaleX, scaleY },
        local,
      );
    }
    item.metadata = [
      ...(item.metadata ?? []),
      { label: 'page', value: pageKey },
    ];
    kept.push(item);
  }
  result.items = kept;
  return { annotation: result, skipped };
}

/**
 * A volume's page-image stems, indexed by their lowercased form.
 *
 * The case of a lettered page suffix is not derivable from the annotation:
 * Chicago's `...-0103W` is `p103w.jpg` on disk while Asheville's `...-0033A` is
 * `p33A.jpg`, so the same URL shape maps to different filenames in different
 * volumes. Rather than guess, callers derive a candidate key and look up the
 * real stem here.
 *
 * Returns an empty map for an unreadable directory, so a caller falls back to
 * its candidate rather than failing.
 */
export async function imageStemsByLowercase(
  volumeDir: string,
): Promise<Map<string, string>> {
  const { readdir } = await import('fs/promises');
  const stems = new Map<string, string>();
  try {
    for (const file of await readdir(volumeDir)) {
      const match = file.match(/^(p[^/]*)\.jpg$/);
      if (match?.[1]) stems.set(match[1].toLowerCase(), match[1]);
    }
  } catch {
    /* no such directory; caller uses its candidate */
  }
  return stems;
}

/** Tile edge advertised in info.json. 512 is the Image API's common default. */
const TILE_WIDTH = 512;

/**
 * Add a `tiles` entry to an Image API info.json body, in place of nothing.
 *
 * express-iiif advertises no tilesets. @allmaps/iiif-parser then synthesises
 * one, and its fallback has an off-by-one:
 *
 *   scaleFactors: Array.from({ length: maxExponent }, (_, e) => 2 ** e)
 *
 * For an image whose longest side is <= its 768 px tile width, maxExponent is
 * 0, so scaleFactors is EMPTY, the map gets zero zoom levels, and rendering
 * dies in getTileImageRequest with "Cannot read properties of undefined
 * (reading 'originalWidth')". 22 small split panels across the truth volumes
 * hit this (grand_rapids p844__3 at 682x568, fargo p62__4 at 365x488).
 *
 * Advertising a real tileset keeps the parser out of that path entirely — and
 * is honest for a level2 service, which serves any region and size. Scale
 * factors always include 1, so the smallest image still gets one zoom level.
 */
export function withTiles(
  info: Record<string, unknown>,
): Record<string, unknown> {
  const width = Number(info.width);
  const height = Number(info.height);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return info;
  if (Array.isArray(info.tiles) && info.tiles.length > 0) return info;
  const maxExponent = Math.ceil(
    Math.log2(Math.max(1, Math.max(width, height) / TILE_WIDTH)),
  );
  const scaleFactors = Array.from(
    { length: Math.max(1, maxExponent + 1) },
    (unused, exponent) => 2 ** exponent,
  );
  return { ...info, tiles: [{ width: TILE_WIDTH, scaleFactors }] };
}
