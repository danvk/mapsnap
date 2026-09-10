import { describe, expect, it } from 'vitest';
import { pngDimensionsFromBuffer } from './pngDimensions';

// The first 24 bytes of a PNG: signature, IHDR length, "IHDR", width, height.
function syntheticPng(width: number, height: number, type = 'IHDR'): Buffer {
  const head = Buffer.alloc(24);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(head, 0);
  head.writeUInt32BE(13, 8);
  head.write(type, 12, 'latin1');
  head.writeUInt32BE(width, 16);
  head.writeUInt32BE(height, 20);
  return head;
}

describe('pngDimensionsFromBuffer', () => {
  it('reads width and height from the IHDR chunk', () => {
    expect(pngDimensionsFromBuffer(syntheticPng(1586, 1949))).toEqual({
      width: 1586,
      height: 1949,
    });
  });

  it('rejects other formats and a misplaced IHDR', () => {
    expect(() =>
      pngDimensionsFromBuffer(Buffer.from([0xff, 0xd8, 0xff, 0xe0])),
    ).toThrow('Not a PNG');
    expect(() => pngDimensionsFromBuffer(syntheticPng(1, 1, 'IDAT'))).toThrow(
      'IHDR',
    );
  });
});
