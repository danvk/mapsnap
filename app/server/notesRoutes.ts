/**
 * Per-page notes API.
 *
 * `registerNotesApi` registers the typed notes endpoints (`/notes-api/*`) on the
 * shared crosswalk router; the path math and validation live in ./notes.
 */

import { mkdir, readFile, readdir, rm, writeFile } from 'fs/promises';
import { HTTPError, type TypedRouter } from 'crosswalk';
import type { API } from './api.ts';
import {
  noteFilePath,
  noteTextOf,
  notesDir,
  pageFromNoteFilename,
} from './notes.ts';

// Page keys with a note sidecar under one volume (empty on error).
async function volumeNotePages(
  dataDir: string,
  volume: string,
): Promise<string[]> {
  try {
    const files = await readdir(notesDir(dataDir, volume));
    return files
      .map(pageFromNoteFilename)
      .filter((page): page is string => page !== null);
  } catch {
    return [];
  }
}

/** Register the typed per-page notes API (`/notes-api/*`). */
export function registerNotesApi(
  router: TypedRouter<API>,
  dataDir: string,
): void {
  // The page keys with a note under one volume, for the volume viewer's markers
  // and tooltip. ?volume=<dir> → { notes: { "<page>": "<text>" } }.
  router.get('/notes-api/notes', async (_params, request) => {
    const { volume } = request.query;
    const notes: Record<string, string> = {};
    for (const page of await volumeNotePages(dataDir, volume)) {
      const path = noteFilePath(dataDir, volume, page);
      if (!path) continue;
      try {
        const text = noteTextOf(JSON.parse(await readFile(path, 'utf8')));
        if (text !== null) notes[page] = text;
      } catch {
        // Unreadable/blank sidecar — skip it.
      }
    }
    return { notes };
  });

  // One page's note. ?volume=<dir>&page=<key> → { note: string }.
  router.get('/notes-api/note', async (_params, request) => {
    const { volume, page } = request.query;
    const path = noteFilePath(dataDir, volume, page);
    if (!path) throw new HTTPError(400, 'bad volume/page');
    try {
      const text = noteTextOf(JSON.parse(await readFile(path, 'utf8')));
      return { note: text ?? '' };
    } catch {
      return { note: '' };
    }
  });

  // Write (or, when blank, delete) one page's note. Body: { note: string }.
  router.put('/notes-api/note', async (_params, body, request) => {
    const { volume, page } = request.query;
    const path = noteFilePath(dataDir, volume, page);
    if (!path) throw new HTTPError(400, 'bad volume/page');
    const note = body.note ?? '';
    if (note.trim() === '') {
      await rm(path, { force: true });
      return { ok: true, note: '' };
    }
    await mkdir(notesDir(dataDir, volume), { recursive: true });
    const stored = { note, updated: new Date().toISOString() };
    await writeFile(path, JSON.stringify(stored, null, 2) + '\n');
    return { ok: true, note };
  });
}
