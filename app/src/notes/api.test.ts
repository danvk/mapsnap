import { describe, expect, it } from 'vitest';
import { noteContextFromFiles } from './api';

describe('noteContextFromFiles', () => {
  it('derives volume and page from a data/ file entry', () => {
    expect(
      noteContextFromFiles([
        'data/los_angeles_ca_1949_vol_14/p1401__2.jpg',
        'data/los_angeles_ca_1949_vol_14/p1401__2.streets.json',
      ]),
    ).toEqual({ volume: 'los_angeles_ca_1949_vol_14', page: 'p1401__2' });
  });

  it('strips every extension to get the page key', () => {
    expect(noteContextFromFiles(['data/vol/p50n.2048px.jpg'])).toEqual({
      volume: 'vol',
      page: 'p50n',
    });
  });

  it('returns null for absolute URLs and non-data paths', () => {
    expect(noteContextFromFiles(['https://example.com/p1.jpg'])).toBeNull();
    expect(noteContextFromFiles(['/tmp/p1.jpg'])).toBeNull();
    expect(noteContextFromFiles([])).toBeNull();
  });

  it('ignores deeper nesting that has no clear volume/page', () => {
    expect(noteContextFromFiles(['data/vol/sub/p1.jpg'])).toBeNull();
  });
});
