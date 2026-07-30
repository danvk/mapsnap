import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
import type { ImageInfo } from '../keymap/types';
import type { KeymapDetection, RegionPolygon } from './types';
import { createRegionsJson, openRing } from './polygons';

const api = typedApi<API>({ fetch: jsonFetch });

/** A sheet's regions with the sheet dimensions its coordinates are in. */
export interface LoadedRegions {
  width: number;
  height: number;
  regions: RegionPolygon[];
}

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

/** Key-map sheets available to trace, reusing the point labeler's page list. */
export async function fetchImages(): Promise<ImageInfo[]> {
  const { images } = await api.get('/api/images')();
  return images;
}

/** Fetch a sheet's hand-drawn regions, or null if none exist yet. */
export async function fetchRegions(id: string): Promise<LoadedRegions | null> {
  const data = await api.get('/api/regions')(null, { id });
  if ('exists' in data) return null;
  return {
    width: data.width,
    height: data.height,
    regions: data.panels.map((ring, index) => ({
      ring: openRing(ring),
      text: data.labels[index] ?? '',
    })),
  };
}

/** Write a sheet's regions in the panels.json schema. */
export async function saveRegions(
  id: string,
  width: number,
  height: number,
  regions: RegionPolygon[],
): Promise<void> {
  await api.put('/api/regions')({}, createRegionsJson(width, height, regions), {
    id,
  });
}

/** Fetch the sheet's detected page numbers, for auto-naming a drawn ring. */
export async function fetchDetections(id: string): Promise<KeymapDetection[]> {
  const { detections } = await api.get('/api/keymap-detections')(null, { id });
  return detections;
}
