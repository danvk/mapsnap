/**
 * Type-safe HTTP API for the debugger app, defined once with crosswalk.
 *
 * This interface is the single source of truth for the JSON API: server.ts
 * serves it with a `TypedRouter<API>` (handlers are checked to return the right
 * shape) and the browser calls it with `typedApi<API>()` (requests are checked
 * against the same types). See https://github.com/danvk/crosswalk.
 *
 * Binary endpoints (the `/iiif` image service, `/api/keymaps/:name` JPEGs, and
 * the `/mapsnap` static build) are not JSON and are served as plain Express
 * middleware in server.ts, so they are not part of this interface.
 */

import type { Endpoint, GetEndpoint } from 'crosswalk/dist/api-spec';
import type {
  RewrittenAnnotationResponse,
  VolumeListResponse,
} from './iiifAnnotations.ts';
import type { ImageInfo, LabelsJson } from '../src/keymap/types.ts';

/** Query naming a georeference AnnotationPage, repo-root-relative. */
export interface AnnotationQuery {
  path: string;
}

/** Response of GET /api/images. */
export interface KeymapImagesResponse {
  images: ImageInfo[];
}

/** GET /api/labels/:name — the sidecar, or a marker that none exists yet. */
export type LabelsResponse = LabelsJson | { exists: false };

/** Response of PUT /api/labels/:name. */
export interface LabelsWriteResponse {
  ok: boolean;
}

/** Query naming a volume directory. */
export interface VolumeQuery {
  volume: string;
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
  '/api/images': {
    get: GetEndpoint<KeymapImagesResponse>;
  };
  '/api/labels/:name': {
    get: GetEndpoint<LabelsResponse>;
    put: Endpoint<LabelsJson, LabelsWriteResponse>;
  };
  '/notes-api/notes': {
    get: GetEndpoint<NotesResponse, VolumeQuery>;
  };
  '/notes-api/note': {
    get: GetEndpoint<NoteResponse, NoteTarget>;
    put: Endpoint<NoteWriteRequest, NoteWriteResponse, NoteTarget>;
  };
}
