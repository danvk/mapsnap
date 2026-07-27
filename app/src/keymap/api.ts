import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
import type { ImageInfo, LabelsJson, LabelsWriteRequest } from './types';

const api = typedApi<API>({ fetch: jsonFetch });

/**
 * URL the server serves a key map image from (a binary, non-JSON endpoint).
 *
 * Page ids contain slashes ("queens_1950/vol2/p0") and the endpoint is a
 * wildcard path, so each segment is encoded separately to keep them.
 */
export function imageUrl(id: string): string {
  const path = id.split('/').map(encodeURIComponent).join('/');
  return `/api/keymaps/${path}`;
}

/** Fetch the list of available key map pages with their label counts. */
export async function fetchImages(): Promise<ImageInfo[]> {
  const { images } = await api.get('/api/images')();
  return images;
}

/** Fetch a page's labels sidecar, or null if none exists yet. */
export async function fetchLabels(id: string): Promise<LabelsJson | null> {
  const data = await api.get('/api/labels')(null, { id });
  return 'exists' in data ? null : data;
}

/** Write a page's labels sidecar. */
export async function saveLabels(
  id: string,
  data: LabelsWriteRequest,
): Promise<void> {
  await api.put('/api/labels')({}, data, { id });
}
