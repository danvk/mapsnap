import type { ImageInfo, LabelsJson } from './types';

const API_BASE = '/api';

/** URL the server serves a key map image from. */
export function imageUrl(name: string): string {
  return `${API_BASE}/keymaps/${encodeURIComponent(name)}`;
}

/** Fetch the list of available key map images with their label counts. */
export async function fetchImages(): Promise<ImageInfo[]> {
  const resp = await fetch(`${API_BASE}/images`);
  if (!resp.ok) throw new Error(`Failed to list images: ${resp.status}`);
  const data = (await resp.json()) as { images: ImageInfo[] };
  return data.images;
}

/** Fetch an image's labels sidecar, or null if none exists yet. */
export async function fetchLabels(name: string): Promise<LabelsJson | null> {
  const resp = await fetch(`${API_BASE}/labels/${encodeURIComponent(name)}`);
  if (!resp.ok) throw new Error(`Failed to load labels: ${resp.status}`);
  const data = (await resp.json()) as LabelsJson | { exists: false };
  if ('exists' in data && data.exists === false) return null;
  return data as LabelsJson;
}

/** Write an image's labels sidecar. */
export async function saveLabels(
  name: string,
  data: LabelsJson,
): Promise<void> {
  const resp = await fetch(`${API_BASE}/labels/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!resp.ok) throw new Error(`Failed to save labels: ${resp.status}`);
}
