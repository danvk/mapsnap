import { describe, expect, it } from 'vitest';

import {
  cachePathOf,
  imagePrefixOf,
  isS3Uri,
  parseS3Uri,
  uriFromCacheRelative,
} from './s3Objects.ts';

describe('parseS3Uri', () => {
  it('splits a bucket from its key', () => {
    expect(
      parseS3Uri('s3://mapsnap-sanborn/by-state/illinois/1950/x.json'),
    ).toEqual({
      bucket: 'mapsnap-sanborn',
      key: 'by-state/illinois/1950/x.json',
    });
  });

  it('tolerates surrounding whitespace, which a pasted URL carries', () => {
    expect(parseS3Uri('  s3://bucket/key  ')?.key).toBe('key');
  });

  it('rejects anything that is not an object reference', () => {
    expect(parseS3Uri('data/detroit/main.iiif.json')).toBeNull();
    expect(parseS3Uri('s3://bucket')).toBeNull();
    expect(parseS3Uri('s3://bucket/')).toBeNull();
    expect(parseS3Uri('https://example.com/x')).toBeNull();
  });

  it('refuses a key that climbs out of its prefix', () => {
    expect(parseS3Uri('s3://bucket/../etc/passwd')).toBeNull();
  });
});

describe('isS3Uri', () => {
  it('separates object references from paths under data/', () => {
    expect(isS3Uri('s3://bucket/key')).toBe(true);
    expect(isS3Uri('  s3://bucket/key')).toBe(true);
    expect(isS3Uri('data/detroit_mich_1929_vol_11/main.iiif.json')).toBe(false);
  });
});

describe('imagePrefixOf', () => {
  it('strips the run directory, where no images live', () => {
    expect(
      imagePrefixOf(
        'by-state/illinois/1950/sanborn01790_090/runs/corpus-v1/mapsnap.iiif.json',
      ),
    ).toBe('by-state/illinois/1950/sanborn01790_090');
  });

  it('leaves an annotation that already sits beside its images', () => {
    expect(
      imagePrefixOf('by-state/illinois/1950/sanborn01790_090/main.iiif.json'),
    ).toBe('by-state/illinois/1950/sanborn01790_090');
  });

  it('only strips a real runs/<tag> pair', () => {
    // A directory called "runs" with the annotation directly inside it is not
    // the layout loc-fit writes, and taking two segments off would climb into
    // the wrong item.
    expect(
      imagePrefixOf('by-state/ohio/1950/item/runs/mapsnap.iiif.json'),
    ).toBe('by-state/ohio/1950/item/runs');
  });
});

describe('cachePathOf and uriFromCacheRelative', () => {
  it('round-trips an object through its cache path', () => {
    const uri = {
      bucket: 'mapsnap-sanborn',
      key: 'by-state/ohio/1950/item/p1.jpg',
    };
    const cached = cachePathOf('/cache', uri);
    expect(cached).toBe(
      '/cache/mapsnap-sanborn/by-state/ohio/1950/item/p1.jpg',
    );
    expect(
      uriFromCacheRelative('mapsnap-sanborn/by-state/ohio/1950/item/p1.jpg'),
    ).toEqual(uri);
  });

  it('refuses a relative path that escapes the cache', () => {
    expect(uriFromCacheRelative('../../etc/passwd')).toBeNull();
    expect(uriFromCacheRelative('bucket-only')).toBeNull();
  });
});
