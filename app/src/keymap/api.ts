import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
import type { ImageInfo, LabelsJson } from './types';

const api = typedApi<API>({ fetch: jsonFetch });

/** URL the server serves a key map image from (a binary, non-JSON endpoint). */
export function imageUrl(name: string): string {
  return `/api/keymaps/${encodeURIComponent(name)}`;
}

/** Fetch the list of available key map images with their label counts. */
export async function fetchImages(): Promise<ImageInfo[]> {
  const { images } = await api.get('/api/images')();
  return images;
}

/** Fetch an image's labels sidecar, or null if none exists yet. */
export async function fetchLabels(name: string): Promise<LabelsJson | null> {
  const data = await api.get('/api/labels/:name')({ name });
  return 'exists' in data ? null : data;
}

/** Write an image's labels sidecar. */
export async function saveLabels(
  name: string,
  data: LabelsJson,
): Promise<void> {
  await api.put('/api/labels/:name')({ name }, data);
}
