/**
 * Type-safe HTTP API for the debugger app, defined once with crosswalk.
 *
 * This interface is the single source of truth for the JSON API: server.ts
 * serves it with a `TypedRouter<API>` (handlers are checked to return the right
 * shape) and the browser calls it with `typedApi<API>()` (requests are checked
 * against the same types). See https://github.com/danvk/crosswalk.
 *
 * Binary endpoints (the `/iiif` image service, `/api/keymaps/*` key-map images,
 * and the `/mapsnap` static build) are not JSON and are served as plain Express
 * middleware in server.ts, so they are not part of this interface.
 */

import type { Endpoint, GetEndpoint } from 'crosswalk/dist/api-spec';
import type {
  RewrittenAnnotationResponse,
  VolumeListResponse,
} from './iiifAnnotations.ts';
import type { CompareResponse } from './compareTxt.ts';
import type {
  ImageInfo,
  LabelsJson,
  LabelsWriteRequest,
} from '../src/keymap/types.ts';
import type {
  KeymapDetection,
  RegionsJson,
  RegionsWriteRequest,
} from '../src/regions/types.ts';
import type { AdjacencyData } from '../src/types.ts';

/** Response of GET /iiif-api/adjacency — the volume's adjacency.json, or null when absent. */
export interface AdjacencyResponse {
  adjacency: AdjacencyData | null;
}

/**
 * Response of GET /iiif-api/osm-relation — the boundary of the OSM relation the
 * volume's street network was downloaded from, as one LineString per member way.
 *
 * `null` when the volume has no r<id>.json. The ways are NOT stitched into a
 * ring: drawing them as separate lines traces the same boundary and avoids
 * reimplementing polygon assembly in the browser.
 */
export interface OsmRelationResponse {
  relation: {
    id: string;
    name: string | null;
    ways: [number, number][][];
  } | null;
}

/** Query naming a georeference AnnotationPage, repo-root-relative. */
export interface AnnotationQuery {
  path: string;
}

/** Response of GET /api/images. */
export interface KeymapImagesResponse {
  images: ImageInfo[];
}

/** Query naming one key-map page, e.g. `{ id: "queens_1950/vol2/p0" }`. */
export interface KeymapTarget {
  id: string;
}

/** GET /api/labels — the sidecar, or a marker that none exists yet. */
export type LabelsResponse = LabelsJson | { exists: false };

/** Response of PUT /api/labels. */
export interface LabelsWriteResponse {
  ok: boolean;
}

/** GET /api/regions — the hand-drawn page regions, or a marker that none exist yet. */
export type RegionsResponse = RegionsJson | { exists: false };

/**
 * GET /api/keymap-detections — the sheet's detected page numbers as points.
 *
 * The region labeler uses these to name a freshly drawn ring by whichever number it
 * encloses, so a sheet can be traced without typing 40 page keys. Empty when the sheet
 * has no `<stem>.keymap.json`.
 */
export interface KeymapDetectionsResponse {
  detections: KeymapDetection[];
}

/** Query naming a volume directory. */
export interface VolumeQuery {
  volume: string;
}

/** Response of GET /api/adjacency-volumes. */
export interface AdjacencyVolumesResponse {
  volumes: { name: string; pageCount: number; labeledPages: number }[];
}

/** Response of GET /api/adjacency-pages. */
export interface AdjacencyPagesResponse {
  pages: ImageInfo[];
}

/** Query naming one page of one volume for adjacency truth. */
export interface AdjacencyTruthTarget {
  volume: string;
  page: string;
}

/** GET /api/adjacency-labels — the page's entry, or a marker that none exists. */
export type AdjacencyLabelsResponse = LabelsWriteRequest | { exists: false };

/**
 * Response of GET /iiif-api/failed-georefs — a volume's page files.
 *
 * `failed` maps each page that has a failed-georef sidecar (`<stem>.georef-<kind>.json`,
 * e.g. `p1452.georef-nofit.json`) to its kind ("nofit", "1gcp", "misscale", …),
 * so the volume viewer can link a not-georeferenced page to that debug file.
 *
 * `pages` is every page-image stem in the volume, split sheets represented by their
 * panels. A volume with no truth annotation has no other way to know which pages
 * exist, so this is what the viewer subtracts the fitted pages from to list the
 * un-fit ones. Optional so a client talking to an older server degrades to listing
 * none rather than failing.
 */
export interface FailedGeorefsResponse {
  /**
   * Page stem → every `<stem>.georef*.json` sidecar it has, sorted.
   *
   * Was a single "failure kind" per stem, decoded from the filename. #270
   * phase 3 collapsed those suffixes into a `status` field inside one sidecar
   * per channel, so the name no longer says anything about failure -- and the
   * old pattern started matching `-final`/`-snap`/`-street`, i.e. every page.
   * The viewer lists what is there and lets each file speak for itself (#252).
   */
  georefs: Record<string, string[]>;
  pages?: string[];
}

/** One key-map sheet in a volume's `raw/` directory and which sidecars it has. */
export interface KeymapInfo {
  /** Key-map image stem, e.g. "p0" (has a `raw/<stem>.keymap.json`). */
  stem: string;
  /** Whether a `raw/<stem>.regions.panels.json` region-segmentation sidecar exists. */
  hasRegions: boolean;
  /** Whether a `raw/<stem>.georef.json` sidecar exists. */
  hasGeoref: boolean;
}

/**
 * Response of GET /iiif-api/run-artifacts — where a run's own sidecars live.
 *
 * A tagged run may keep per-page sidecars under `artifacts/<tag>/`, in which
 * case the viewer should link there: the top-level `<stem>.georef.json` is
 * whatever the last run wrote and need not be what produced the annotation
 * being looked at. Not every run saves them, so this reports what is actually
 * present rather than assuming.
 *
 * `dir` is the artifact directory, repo-root-relative, or null when the run has
 * none. `stems` lists the page stems it holds a sidecar for, so the viewer can
 * fall back per page instead of all-or-nothing — a run that saved only its
 * failures still links those correctly.
 */
export interface RunArtifactsResponse {
  dir: string | null;
  stems: string[];
}

/** Response of GET /iiif-api/keymaps — a volume's key-map sheets, for the info-panel links. */
export interface KeymapsResponse {
  keymaps: KeymapInfo[];
}

/** Query naming one page of one volume. */
export interface NoteTarget {
  volume: string;
  page: string;
}

/** Response of GET /notes-api/notes — page key → note text. */
export interface NotesResponse {
  notes: Record<string, string>;
}

/** Response of GET /notes-api/note. */
export interface NoteResponse {
  note: string;
}

/** Body of PUT /notes-api/note. */
export interface NoteWriteRequest {
  note: string;
}

/** Response of PUT /notes-api/note (echoes the stored text; "" if deleted). */
export interface NoteWriteResponse {
  ok: boolean;
  note: string;
}

export interface API {
  '/iiif-api/volumes': {
    get: GetEndpoint<VolumeListResponse>;
  };
  '/iiif-api/annotation': {
    get: GetEndpoint<RewrittenAnnotationResponse, AnnotationQuery>;
  };
  '/iiif-api/failed-georefs': {
    get: GetEndpoint<FailedGeorefsResponse, VolumeQuery>;
  };
  '/iiif-api/compare': {
    get: GetEndpoint<CompareResponse, AnnotationQuery>;
  };
  '/iiif-api/adjacency': {
    get: GetEndpoint<AdjacencyResponse, VolumeQuery>;
  };
  '/iiif-api/osm-relation': {
    get: GetEndpoint<OsmRelationResponse, VolumeQuery>;
  };
  '/iiif-api/run-artifacts': {
    get: GetEndpoint<RunArtifactsResponse, AnnotationQuery>;
  };
  '/iiif-api/keymaps': {
    get: GetEndpoint<KeymapsResponse, VolumeQuery>;
  };
  '/api/images': {
    get: GetEndpoint<KeymapImagesResponse>;
  };
  '/api/labels': {
    get: GetEndpoint<LabelsResponse, KeymapTarget>;
    put: Endpoint<LabelsWriteRequest, LabelsWriteResponse, KeymapTarget>;
  };
  '/api/regions': {
    get: GetEndpoint<RegionsResponse, KeymapTarget>;
    put: Endpoint<RegionsWriteRequest, LabelsWriteResponse, KeymapTarget>;
  };
  '/api/keymap-detections': {
    get: GetEndpoint<KeymapDetectionsResponse, KeymapTarget>;
  };
  '/api/adjacency-volumes': {
    get: GetEndpoint<AdjacencyVolumesResponse>;
  };
  '/api/adjacency-pages': {
    get: GetEndpoint<AdjacencyPagesResponse, VolumeQuery>;
  };
  '/api/adjacency-labels': {
    get: GetEndpoint<AdjacencyLabelsResponse, AdjacencyTruthTarget>;
    put: Endpoint<
      LabelsWriteRequest,
      LabelsWriteResponse,
      AdjacencyTruthTarget
    >;
  };
  '/notes-api/notes': {
    get: GetEndpoint<NotesResponse, VolumeQuery>;
  };
  '/notes-api/note': {
    get: GetEndpoint<NoteResponse, NoteTarget>;
    put: Endpoint<NoteWriteRequest, NoteWriteResponse, NoteTarget>;
  };
}
