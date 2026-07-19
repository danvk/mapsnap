import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
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
