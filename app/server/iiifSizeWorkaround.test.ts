import { describe, expect, it } from 'vitest';

import { normalizeIiifImageUrl } from './iiifSizeWorkaround';

describe('normalizeIiifImageUrl', () => {
  it('rewrites identity-size tile requests to max', () => {
    expect(
      normalizeIiifImageUrl(
        '/vol/p220.jpg/768,0,655,768/655,768/0/default.jpg',
      ),
    ).toBe('/vol/p220.jpg/768,0,655,768/max/0/default.jpg');
  });

  it('leaves genuine downscales alone', () => {
    const url = '/vol/p220.jpg/0,0,1423,1536/712,768/0/default.jpg';
    expect(normalizeIiifImageUrl(url)).toBe(url);
  });

  it('leaves non-region requests alone', () => {
    const url = '/vol/p220.jpg/full/356,/0/default.jpg';
    expect(normalizeIiifImageUrl(url)).toBe(url);
    expect(normalizeIiifImageUrl('/vol/p220.jpg/info.json')).toBe(
      '/vol/p220.jpg/info.json',
    );
  });

  it('only rewrites when both dimensions match', () => {
    const url = '/vol/p220.jpg/768,0,655,768/655,700/0/default.jpg';
    expect(normalizeIiifImageUrl(url)).toBe(url);
  });
});
