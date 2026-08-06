import { describe, expect, it } from 'vitest';
import { parseAnnotationPath } from './volumePath';

describe('parseAnnotationPath', () => {
  it('splits a single-directory volume', () => {
    expect(
      parseAnnotationPath(
        'data/detroit_mich_1929_vol_11/2026-08-05-base.iiif.json',
      ),
    ).toEqual({
      volume: 'detroit_mich_1929_vol_11',
      file: '2026-08-05-base.iiif.json',
    });
  });

  it('splits a subvolume of a multi-volume atlas (#228)', () => {
    expect(
      parseAnnotationPath('data/brooklyn_1904-1908/vol13/2026-08-04.iiif.json'),
    ).toEqual({
      volume: 'brooklyn_1904-1908/vol13',
      file: '2026-08-04.iiif.json',
    });
  });

  it('does not parse a path deeper than the server would accept', () => {
    // Mirrors MAX_VOLUME_DEPTH: parsing this would have the UI query a volume
    // every volume-scoped endpoint rejects.
    expect(parseAnnotationPath('data/a/b/c/x.iiif.json')).toBeNull();
  });

  it('returns null for a non-data or incomplete path', () => {
    expect(parseAnnotationPath(null)).toBeNull();
    expect(parseAnnotationPath('data/x.iiif.json')).toBeNull();
    expect(parseAnnotationPath('other/vol/x.iiif.json')).toBeNull();
  });
});
