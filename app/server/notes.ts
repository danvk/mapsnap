/**
 * Per-page note sidecars for the debugger app.
 *
 * A note is a free-text scratchpad attached to one page of one volume, stored
 * at `data/<volume>/artifacts/notes/<page>.json`. These helpers are pure (path
 * math + validation only); server.mjs does the filesystem work.
 */

import { join } from 'path';

/** On-disk shape of a note sidecar. */
export interface NoteFile {
  note: string;
  /** ISO-8601 timestamp of the last write. */
  updated: string;
}

// A path segment safe to join under the data directory: non-empty, no slashes
// or parent-directory escapes. (Same rule the image/label servers use.)
function isSafeSegment(name: string): boolean {
  return (
    name !== '' &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..')
  );
}

/**
 * Whether ``volume`` and ``page`` are safe to use as path segments.
 *
 * ``page`` must additionally be a bare page key (letters, digits, underscores —
 * e.g. "p1401__2"), so the only file it can name is ``<page>.json``.
 */
export function isValidNoteTarget(volume: string, page: string): boolean {
  return isSafeSegment(volume) && /^[A-Za-z0-9_]+$/.test(page);
}

/** Directory holding a volume's note sidecars, under ``dataDir``. */
export function notesDir(dataDir: string, volume: string): string {
  return join(dataDir, volume, 'artifacts', 'notes');
}

/** Absolute path of one page's note sidecar, or null if the target is unsafe. */
export function noteFilePath(
  dataDir: string,
  volume: string,
  page: string,
): string | null {
  if (!isValidNoteTarget(volume, page)) return null;
  return join(notesDir(dataDir, volume), `${page}.json`);
}

/** Page key from a note sidecar filename, or null if it isn't a ``.json``. */
export function pageFromNoteFilename(filename: string): string | null {
  const match = filename.match(/^([A-Za-z0-9_]+)\.json$/);
  return match ? (match[1] ?? null) : null;
}

/**
 * The note text of a parsed sidecar, or null when the object isn't a note.
 *
 * Tolerates a bare string or a legacy `{ text }` field so an older sidecar
 * still reads, but a blank note is treated as absent (the writer deletes it).
 */
export function noteTextOf(parsed: unknown): string | null {
  if (typeof parsed === 'string') return parsed.trim() ? parsed : null;
  if (parsed && typeof parsed === 'object') {
    const value =
      (parsed as { note?: unknown }).note ??
      (parsed as { text?: unknown }).text;
    if (typeof value === 'string') return value.trim() ? value : null;
  }
  return null;
}
