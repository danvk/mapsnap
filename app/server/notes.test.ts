import { describe, expect, it } from 'vitest';
import {
  isValidNoteTarget,
  noteFilePath,
  noteTextOf,
  pageFromNoteFilename,
} from './notes';

describe('isValidNoteTarget', () => {
  it('accepts a real volume and page key, including split panels', () => {
    expect(isValidNoteTarget('los_angeles_ca_1949_vol_14', 'p1401__2')).toBe(
      true,
    );
    expect(isValidNoteTarget('brooklyn_1904-1908', 'p6N')).toBe(true);
  });

  it('rejects path escapes and separators', () => {
    expect(isValidNoteTarget('..', 'p1')).toBe(false);
    expect(isValidNoteTarget('vol/sub', 'p1')).toBe(false);
    expect(isValidNoteTarget('vol', '../secret')).toBe(false);
    expect(isValidNoteTarget('vol', 'p1/p2')).toBe(false);
    expect(isValidNoteTarget('vol', 'p1.json')).toBe(false); // no dots in page
    expect(isValidNoteTarget('', 'p1')).toBe(false);
  });
});

describe('noteFilePath', () => {
  it('builds the artifacts/notes path', () => {
    expect(noteFilePath('/data', 'vol_1', 'p49')).toBe(
      '/data/vol_1/artifacts/notes/p49.json',
    );
  });

  it('returns null for an unsafe target', () => {
    expect(noteFilePath('/data', 'vol', '../x')).toBeNull();
  });
});

describe('pageFromNoteFilename', () => {
  it('extracts the page key from a sidecar filename', () => {
    expect(pageFromNoteFilename('p1401__2.json')).toBe('p1401__2');
    expect(pageFromNoteFilename('p49.json')).toBe('p49');
  });

  it('ignores non-note files', () => {
    expect(pageFromNoteFilename('p49.streets.json')).toBeNull();
    expect(pageFromNoteFilename('notes.txt')).toBeNull();
  });
});

describe('noteTextOf', () => {
  it('reads the note field, a bare string, and the legacy text field', () => {
    expect(noteTextOf({ note: 'hi', updated: 'x' })).toBe('hi');
    expect(noteTextOf('hi')).toBe('hi');
    expect(noteTextOf({ text: 'legacy' })).toBe('legacy');
  });

  it('treats blank or malformed content as absent', () => {
    expect(noteTextOf({ note: '   ' })).toBeNull();
    expect(noteTextOf({})).toBeNull();
    expect(noteTextOf(null)).toBeNull();
    expect(noteTextOf(42)).toBeNull();
  });
});
