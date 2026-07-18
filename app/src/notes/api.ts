/**
 * Client for the per-page notes API (served by server.mjs at /notes-api).
 *
 * A note is free text attached to one page of one volume, identified the same
 * way the debugger's `?files=` deep links are: a volume directory name plus a
 * page key (e.g. "los_angeles_ca_1949_vol_14" / "p1401__2").
 */

import { pageStem } from '../fileLoading';

const API_BASE = '/notes-api';

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
  const resp = await fetch(
    `${API_BASE}/note?volume=${encodeURIComponent(ctx.volume)}&page=${encodeURIComponent(ctx.page)}`,
  );
  if (!resp.ok) throw new Error(`Failed to load note: ${resp.status}`);
  return ((await resp.json()) as { note: string }).note;
}

/** Save one page's note; a blank note deletes it. Returns the stored text. */
export async function saveNote(
  ctx: NoteContext,
  note: string,
): Promise<string> {
  const resp = await fetch(
    `${API_BASE}/note?volume=${encodeURIComponent(ctx.volume)}&page=${encodeURIComponent(ctx.page)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note }),
    },
  );
  if (!resp.ok) throw new Error(`Failed to save note: ${resp.status}`);
  return ((await resp.json()) as { note: string }).note;
}

/** Fetch every page's note in a volume, as a page-key → text map. */
export async function fetchVolumeNotes(
  volume: string,
): Promise<Map<string, string>> {
  const resp = await fetch(
    `${API_BASE}/notes?volume=${encodeURIComponent(volume)}`,
  );
  if (!resp.ok) throw new Error(`Failed to load notes: ${resp.status}`);
  const data = (await resp.json()) as { notes: Record<string, string> };
  return new Map(Object.entries(data.notes));
}
