import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
import type { ComparePageStats } from '../../server/compareTxt';
import type {
  RewrittenAnnotationResponse,
  VolumeListResponse,
} from '../../server/iiifAnnotations';

const api = typedApi<API>({ fetch: jsonFetch });

/** Fetch the volumes that have local page images and georeference annotations. */
export function fetchVolumes(): Promise<VolumeListResponse> {
  return api.get('/iiif-api/volumes')();
}

/**
 * Fetch an annotation file, rewritten to target the local IIIF image server.
 *
 * The path is repo-root-relative, e.g. "data/brooklyn_ny_1906_vol_6/generated.iiif.json".
 */
export function fetchRewrittenAnnotation(
  path: string,
): Promise<RewrittenAnnotationResponse> {
  return api.get('/iiif-api/annotation')(null, { path });
}

/**
 * Fetch a volume's failed-georef sidecars as a page-stem → failure-kind map
 * (e.g. "p1452" → "nofit"), for linking un-georeferenced pages to the georef view.
 */
export async function fetchFailedGeorefs(
  volume: string,
): Promise<Map<string, string>> {
  const { failed } = await api.get('/iiif-api/failed-georefs')(null, {
    volume,
  });
  return new Map(Object.entries(failed));
}

/**
 * Fetch the per-page truth comparison from an annotation's `mapsnap compare` sidecar table,
 * as paired-page stats keyed by generated page stem. Empty when there is no sidecar.
 */
export async function fetchCompare(path: string): Promise<ComparePageStats[]> {
  const { pages } = await api.get('/iiif-api/compare')(null, { path });
  return pages;
}
