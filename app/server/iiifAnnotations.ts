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
 * Extract the page key from a LOC IIIF service URL, or null for non-page images.
 *
 * Port of _service_url_to_page_key in mapsnap/make_iiif_georef.py. The page key
 * is the trailing segment after the last "-", with leading zeros stripped and
 * any letter suffix lowercased:
 *   "...:01790_01N_1950-0006N/info.json" → "p6n"
 *   "...:05791_02_1939-0027s"            → "p27s"
 * Sanborn sb-format (5 digits then a suffix char, '0' meaning none):
 *   "...:sb001250" → "p125";  "...:sb00154s" → "p154s"
 * Covers and indexes ("...-covr", "...-titl") and missing URLs return null.
 */
export function serviceUrlToPageKey(
  url: string | null | undefined,
): string | null {
  if (!url) return null;
  url = url.replace(/\/info\.json$/, '');
  const sbMatch = url.match(/:sb(\d{5})([a-z0-9])$/i);
  if (sbMatch) {
    const pageNum = parseInt(sbMatch[1] ?? '', 10);
    const suffixChar = (sbMatch[2] ?? '').toLowerCase();
    const suffix = suffixChar === '0' ? '' : suffixChar;
    return `p${pageNum}${suffix}`;
  }
  const match = url.match(/-0*(\d+)([a-z]*)$/i);
  if (!match) return null;
  return `p${match[1]}${(match[2] ?? '').toLowerCase()}`;
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
): RewriteResult {
  const result: GeorefAnnotationPage = structuredClone(page);
  const kept: GeorefAnnotationItem[] = [];
  const skipped: SkippedItem[] = [];
  for (const item of result.items ?? []) {
    const label = String(item.label ?? item.id ?? '');
    const target = item.target;
    const source = target?.source;
    const pageKey = serviceUrlToPageKey(source?.id);
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
