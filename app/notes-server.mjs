/**
 * Per-page notes API, mounted on a shared Express app.
 *
 * A note is free text attached to one page of one volume, stored at
 * <data>/<volume>/artifacts/notes/<page>.json as { note, updated }:
 *   GET /notes-api/notes?volume=       →  { notes: { "<page>": "<text>" } }
 *   GET /notes-api/note?volume=&page=  →  { note: string }
 *   PUT /notes-api/note?volume=&page=  →  write it ({ note }); a blank note deletes it
 *
 * registerNotesRoutes(app, { dataDir }) wires these onto the app; server.mjs owns
 * the process. The path math and validation live in ./server/notes.ts.
 */

import { mkdir, readFile, readdir, rm, writeFile } from 'fs/promises';
import {
  noteFilePath,
  noteTextOf,
  notesDir,
  pageFromNoteFilename,
} from './server/notes.ts';

// Reject a volume name that could escape the data directory.
function isSafeName(name) {
  return (
    name !== '' &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..')
  );
}

// Page keys with a note sidecar under one volume, as a Set (empty on error).
async function volumeNotePages(dataDir, volume) {
  try {
    const files = await readdir(notesDir(dataDir, volume));
    const pages = files
      .map(pageFromNoteFilename)
      .filter((page) => page !== null);
    return new Set(pages);
  } catch {
    return new Set();
  }
}

/** Mount the per-page notes API on ``app``. */
export function registerNotesRoutes(app, { dataDir }) {
  // The page keys with a note under one volume, for the volume viewer's markers
  // and tooltip. ?volume=<dir> → { notes: { "<page>": "<text>" } }.
  app.get('/notes-api/notes', async (req, res) => {
    const volume = String(req.query.volume ?? '');
    if (!isSafeName(volume))
      return res.status(400).json({ error: 'bad volume' });
    const notes = {};
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
    res.json({ notes });
  });

  // One page's note. ?volume=<dir>&page=<key> → { note: string }.
  app.get('/notes-api/note', async (req, res) => {
    const path = noteFilePath(
      dataDir,
      String(req.query.volume ?? ''),
      String(req.query.page ?? ''),
    );
    if (!path) return res.status(400).json({ error: 'bad volume/page' });
    try {
      const text = noteTextOf(JSON.parse(await readFile(path, 'utf8')));
      res.json({ note: text ?? '' });
    } catch {
      res.json({ note: '' });
    }
  });

  // Write (or, when blank, delete) one page's note. Body: { note: string }.
  app.put('/notes-api/note', async (req, res) => {
    const volume = String(req.query.volume ?? '');
    const page = String(req.query.page ?? '');
    const path = noteFilePath(dataDir, volume, page);
    if (!path) return res.status(400).json({ error: 'bad volume/page' });
    const note = String(req.body?.note ?? '');
    try {
      if (note.trim() === '') {
        await rm(path, { force: true });
      } else {
        await mkdir(notesDir(dataDir, volume), { recursive: true });
        const body = { note, updated: new Date().toISOString() };
        await writeFile(path, JSON.stringify(body, null, 2) + '\n');
      }
      res.json({ ok: true, note: note.trim() === '' ? '' : note });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });
}
