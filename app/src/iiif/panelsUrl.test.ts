import { describe, expect, it } from 'vitest';

import { panelsUrlFor } from './panelsUrl.ts';

const S3 =
  's3://mapsnap-sanborn/by-state/wisconsin/1910/sanborn09554_005/runs/batch-test-200/mapsnap.iiif.json';

describe('panelsUrlFor', () => {
  it('names the sidecar beside an annotation in the mirror', () => {
    const url = panelsUrlFor(S3, undefined, 'p2');
    expect(url).toBe(
      '/s3-api/object?uri=' +
        encodeURIComponent(
          's3://mapsnap-sanborn/by-state/wisconsin/1910/sanborn09554_005/runs/batch-test-200/p2.panels.json',
        ),
    );
    // Only the object name changes; the run tag has to survive.
    expect(decodeURIComponent(url ?? '')).toContain('/runs/batch-test-200/');
  });

  it('reads a local volume out of data/', () => {
    expect(
      panelsUrlFor('data/fargo_nd_1958/main.iiif.json', 'fargo_nd_1958', 'p45'),
    ).toBe('/data/fargo_nd_1958/p45.panels.json');
  });

  it('prefers the object path even when a volume name was parsed', () => {
    expect(panelsUrlFor(S3, 'somehow_parsed', 'p2')).toContain(
      '/s3-api/object',
    );
  });

  it('has nowhere to look without a volume or an object', () => {
    expect(panelsUrlFor('data/x.iiif.json', undefined, 'p1')).toBeNull();
    expect(panelsUrlFor(null, undefined, 'p1')).toBeNull();
    expect(panelsUrlFor(S3, undefined, '')).toBeNull();
  });
});
