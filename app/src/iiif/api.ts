import type {
  RewrittenAnnotationResponse,
  VolumeListResponse,
} from '../../server/iiifAnnotations';

const API_BASE = '/iiif-api';

/** Fetch the volumes that have local page images and georeference annotations. */
export async function fetchVolumes(): Promise<VolumeListResponse> {
  const resp = await fetch(`${API_BASE}/volumes`);
  if (!resp.ok) throw new Error(`Failed to list volumes: ${resp.status}`);
  return (await resp.json()) as VolumeListResponse;
}

/**
 * Fetch an annotation file, rewritten to target the local IIIF image server.
 *
 * The path is repo-root-relative, e.g. "data/brooklyn_ny_1906_vol_6/generated.iiif.json".
 */
export async function fetchRewrittenAnnotation(
  path: string,
): Promise<RewrittenAnnotationResponse> {
  const resp = await fetch(
    `${API_BASE}/annotation?path=${encodeURIComponent(path)}`,
  );
  if (!resp.ok) throw new Error(`Failed to load annotation: ${resp.status}`);
  return (await resp.json()) as RewrittenAnnotationResponse;
}
