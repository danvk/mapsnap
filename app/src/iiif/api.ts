import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API, KeymapInfo, RunArtifactsResponse } from '../../server/api';
import type { CompareResponse } from '../../server/compareTxt';
import type { OsmRelationResponse } from '../../server/api';
import type {
  RewrittenAnnotationResponse,
  VolumeListResponse,
} from '../../server/iiifAnnotations';
import type { AdjacencyData } from '../types';

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

/** A volume's page-image stems, and which of them have a failed-georef sidecar. */
export interface VolumePageFiles {
  /** Every page-image stem, split sheets represented by their panels. */
  stems: string[];
  /** Page stem → failure kind ("nofit", "1gcp", …), for linking to the georef view. */
  failed: Map<string, string>;
}

/**
 * Fetch a volume's page files: the stems on disk plus the failed-georef sidecars.
 *
 * An older server omits `pages`; the stems then come back empty, which costs the
 * un-fit listing on truth-less volumes rather than taking the viewer down.
 */
export async function fetchVolumePageFiles(
  volume: string,
): Promise<VolumePageFiles> {
  const { failed, pages } = await api.get('/iiif-api/failed-georefs')(null, {
    volume,
  });
  return { stems: pages ?? [], failed: new Map(Object.entries(failed)) };
}

/**
 * Fetch the per-page truth comparison from an annotation's `mapsnap compare` sidecar table:
 * paired-page stats keyed by generated page stem, plus the table's summary footer. Pages are
 * empty and footer is "" when there is no sidecar.
 */
export async function fetchCompare(path: string): Promise<CompareResponse> {
  return api.get('/iiif-api/compare')(null, { path });
}

/**
 * Fetch the boundary of the OSM relation the volume's streets came from, or null.
 *
 * Used to show which pages sit outside the downloaded network: their streets are
 * absent from the vocabulary, so they cannot be fit however good the reads are.
 */
export async function fetchOsmRelation(
  volume: string,
): Promise<OsmRelationResponse['relation']> {
  const { relation } = await api.get('/iiif-api/osm-relation')(null, {
    volume,
  });
  return relation;
}

/** Fetch a volume's adjacency.json (per-page sheet-number claims + mutual graph), or null. */
export async function fetchAdjacency(
  volume: string,
): Promise<AdjacencyData | null> {
  const { adjacency } = await api.get('/iiif-api/adjacency')(null, { volume });
  return adjacency;
}

/** Fetch a volume's key-map sheets (raw/*.keymap.json) and which sidecars each has. */
/**
 * Where the run that produced `path` kept its own per-page sidecars.
 *
 * Returns `{ dir: null, stems: [] }` when the run saved none, which is the
 * common case and not an error — the caller then falls back to the volume's
 * top-level sidecars and says so.
 */
export async function fetchRunArtifacts(
  path: string,
): Promise<RunArtifactsResponse> {
  return api.get('/iiif-api/run-artifacts')(null, { path });
}

export async function fetchKeymaps(volume: string): Promise<KeymapInfo[]> {
  const { keymaps } = await api.get('/iiif-api/keymaps')(null, { volume });
  return keymaps;
}
