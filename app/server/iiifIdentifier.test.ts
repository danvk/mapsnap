import { describe, expect, it } from 'vitest';

import { iiifIdentifierOf } from './iiifRoutes.ts';

describe('iiifIdentifierOf', () => {
  it('reads the image out of an info.json request', () => {
    expect(
      iiifIdentifierOf('/bucket/by-state/ohio/1950/item/p1.jpg/info.json'),
    ).toBe('bucket/by-state/ohio/1950/item/p1.jpg');
  });

  it('reads it out of an image request, whatever the parameters', () => {
    expect(
      iiifIdentifierOf('/bucket/item/p1.jpg/full/400,/0/default.jpg'),
    ).toBe('bucket/item/p1.jpg');
    expect(
      iiifIdentifierOf('/bucket/item/p1.jpg/0,0,512,512/256,/90/gray.png'),
    ).toBe('bucket/item/p1.jpg');
  });

  it('declines anything that is not a IIIF request', () => {
    // Too few segments to carry region/size/rotation/quality.
    expect(iiifIdentifierOf('/bucket/p1.jpg')).toBeNull();
    expect(iiifIdentifierOf('/')).toBeNull();
    expect(iiifIdentifierOf('/info.json')).toBeNull();
  });
});
