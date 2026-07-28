import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API, AdjacencyVolumesResponse } from '../../server/api';
import type { ImageInfo, LabelsWriteRequest } from '../keymap/types';

const api = typedApi<API>({ fetch: jsonFetch });

/**
 * URL of a volume-root page image, served by the static data route (the Vite
 * serveDataDir plugin in development, express.static in production). Relative
 * to the app base, so it works at /mapsnap/adjacency.html in both.
 */
export function pageImageUrl(volume: string, page: string): string {
  const path = volume.split('/').map(encodeURIComponent).join('/');
  return `data/${path}/${encodeURIComponent(page)}.jpg`;
}

/** Fetch the volumes that have labelable pages, with labelling progress. */
export async function fetchVolumes(): Promise<
  AdjacencyVolumesResponse['volumes']
> {
  const { volumes } = await api.get('/api/adjacency-volumes')();
  return volumes;
}

/** Fetch one volume's pages with their label counts. */
export async function fetchPages(volume: string): Promise<ImageInfo[]> {
  const { pages } = await api.get('/api/adjacency-pages')(null, { volume });
  return pages;
}

/** Fetch a page's adjacency labels, or null if none exist yet. */
export async function fetchLabels(
  volume: string,
  page: string,
): Promise<LabelsWriteRequest | null> {
  const data = await api.get('/api/adjacency-labels')(null, { volume, page });
  return 'exists' in data ? null : data;
}

/** Write a page's adjacency labels. */
export async function saveLabels(
  volume: string,
  page: string,
  data: LabelsWriteRequest,
): Promise<void> {
  await api.put('/api/adjacency-labels')({}, data, { volume, page });
}
