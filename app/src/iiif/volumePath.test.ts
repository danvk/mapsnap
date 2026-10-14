import { describe, expect, it } from 'vitest';
import {
  annotationName,
  debugImageStem,
  panelsViewFiles,
  parseAnnotationPath,
} from './volumePath';

describe('parseAnnotationPath', () => {
  it('splits a single-directory volume', () => {
    expect(
      parseAnnotationPath(
        'data/detroit_mich_1929_vol_11/2026-08-05-base.iiif.json',
      ),
    ).toEqual({
      volume: 'detroit_mich_1929_vol_11',
      file: '2026-08-05-base.iiif.json',
      run: null,
    });
  });

  it('splits a subvolume of a multi-volume atlas (#228)', () => {
    expect(
      parseAnnotationPath('data/brooklyn_1904-1908/vol13/2026-08-04.iiif.json'),
    ).toEqual({
      volume: 'brooklyn_1904-1908/vol13',
      file: '2026-08-04.iiif.json',
      run: null,
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

describe('parseAnnotationPath with a mirror run', () => {
  it('splits a run annotation into volume, run and file', () => {
    expect(
      parseAnnotationPath(
        'data/wernersville_pa_1914/runs/corpus-v1/mapsnap.iiif.json',
      ),
    ).toEqual({
      volume: 'wernersville_pa_1914',
      run: 'runs/corpus-v1',
      file: 'mapsnap.iiif.json',
    });
  });

  it('allows a run under a multi-volume atlas', () => {
    expect(
      parseAnnotationPath(
        'data/brooklyn_1904-1908/vol13/runs/v2/mapsnap.iiif.json',
      ),
    ).toEqual({
      volume: 'brooklyn_1904-1908/vol13',
      run: 'runs/v2',
      file: 'mapsnap.iiif.json',
    });
  });
});

describe('debugImageStem', () => {
  it("links a split panel's own image when it has one", () => {
    expect(debugImageStem('p2__2', new Set(['p2', 'p2__1', 'p2__2']))).toBe(
      'p2__2',
    );
  });

  it('links the sheet when only the sheet is on disk', () => {
    // A mirror volume: the debugger maps p2.jpg to the panel via p2.panels.json.
    expect(debugImageStem('p2__2', new Set(['p1', 'p2']))).toBe('p2');
  });

  it('keeps the stem for a whole sheet or an unknown image list', () => {
    expect(debugImageStem('p4', new Set(['p4']))).toBe('p4');
    expect(debugImageStem('p2__2', null)).toBe('p2__2');
  });
});

describe('annotationName', () => {
  it('is the file name for an annotation at the volume root', () => {
    expect(
      annotationName(
        parseAnnotationPath('data/woodbury_nj_1886/main.iiif.json'),
      ),
    ).toBe('main.iiif.json');
  });

  it("keeps a run's directory, as the volumes API lists it", () => {
    expect(
      annotationName(
        parseAnnotationPath(
          'data/woodbury_nj_1886/runs/corpus-v1/mapsnap.iiif.json',
        ),
      ),
    ).toBe('runs/corpus-v1/mapsnap.iiif.json');
  });

  it('is null for a path that does not parse', () => {
    expect(
      annotationName(parseAnnotationPath('s3://bucket/x.iiif.json')),
    ).toBeNull();
  });
});

describe('panelsViewFiles', () => {
  it("opens a mirror run's panel on its sheet with that run's panels.json", () => {
    expect(
      panelsViewFiles('welsh_la_1903', 'runs/split-ab-model', 'p2__3'),
    ).toEqual([
      'data/welsh_la_1903/p2.jpg',
      'data/welsh_la_1903/runs/split-ab-model/p2.panels.json',
    ]);
  });

  it("reads the volume root's panels.json for a root annotation", () => {
    expect(panelsViewFiles('fargo_nd_1958', null, 'p9j__1')).toEqual([
      'data/fargo_nd_1958/p9j.jpg',
      'data/fargo_nd_1958/p9j.panels.json',
    ]);
  });

  it('is null for an unsplit page', () => {
    expect(panelsViewFiles('fargo_nd_1958', null, 'p12')).toBeNull();
  });
});
