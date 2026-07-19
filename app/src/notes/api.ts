/**
 * Client for the per-page notes API (served at /notes-api; see server/api.ts).
 *
 * A note is free text attached to one page of one volume, identified the same
 * way the debugger's `?files=` deep links are: a volume directory name plus a
 * page key (e.g. "los_angeles_ca_1949_vol_14" / "p1401__2").
 */

import { typedApi } from 'crosswalk';
import { jsonFetch } from '../apiFetch';
import type { API } from '../../server/api';
import { pageStem } from '../fileLoading';

const api = typedApi<API>({ fetch: jsonFetch });

/** The volume + page a note attaches to. */
export interface NoteContext {
  volume: string;
  page: string;
}

/**
 * The (volume, page) a note attaches to, derived from a `?files=` entry list.
 *
 * Notes only work through the data-directory file interface, so the first entry
 * shaped like `data/<volume>/<file>` wins and its page key is the file's stem
 * ("data/vol/p1401__2.streets.json" → page "p1401__2"). Absolute URLs and
 * dropped blobs (no volume path) yield null — no note button is shown.
 */
export function noteContextFromFiles(files: string[]): NoteContext | null {
  for (const file of files) {
    const match = file.match(/^data\/([^/]+)\/([^/]+)$/);
    if (match) {
      const page = pageStem(match[2]!);
      if (page) return { volume: match[1]!, page };
    }
  }
  return null;
}

/** Fetch one page's note text ("" when there is none). */
export async function fetchNote(ctx: NoteContext): Promise<string> {
  const { note } = await api.get('/notes-api/note')(null, ctx);
  return note;
}

/** Save one page's note; a blank note deletes it. Returns the stored text. */
export async function saveNote(
  ctx: NoteContext,
  note: string,
): Promise<string> {
  const result = await api.put('/notes-api/note')({}, { note }, ctx);
  return result.note;
}

/** Fetch every page's note in a volume, as a page-key → text map. */
export async function fetchVolumeNotes(
  volume: string,
): Promise<Map<string, string>> {
  const { notes } = await api.get('/notes-api/notes')(null, { volume });
  return new Map(Object.entries(notes));
}
