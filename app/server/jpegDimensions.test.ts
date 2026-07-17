import { describe, expect, it } from 'vitest';
import { jpegDimensionsFromBuffer } from './jpegDimensions';

// Build a minimal JPEG: SOI, an APP0 segment, then an SOF marker with the
// given dimensions (payload: precision byte, height, width, component count).
function syntheticJpeg(
  width: number,
  height: number,
  sofMarker = 0xc0,
): Buffer {
  const app0 = Buffer.from([0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46]);
  const sof = Buffer.alloc(11);
  sof[0] = 0xff;
  sof[1] = sofMarker;
  sof.writeUInt16BE(9, 2); // segment length (excludes the marker bytes)
  sof[4] = 8; // precision
  sof.writeUInt16BE(height, 5);
  sof.writeUInt16BE(width, 7);
  sof[9] = 1; // component count
  return Buffer.concat([Buffer.from([0xff, 0xd8]), app0, sof]);
}

describe('jpegDimensionsFromBuffer', () => {
  it('reads dimensions from an SOF0 marker after other segments', () => {
    expect(jpegDimensionsFromBuffer(syntheticJpeg(2048, 2983))).toEqual({
      width: 2048,
      height: 2983,
    });
  });

  it('reads dimensions from a progressive (SOF2) JPEG', () => {
    expect(jpegDimensionsFromBuffer(syntheticJpeg(640, 480, 0xc2))).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('throws on a non-JPEG buffer', () => {
    expect(() => jpegDimensionsFromBuffer(Buffer.from('PNG-ish'))).toThrow(
      'Not a JPEG',
    );
  });

  it('throws when no SOF marker is present', () => {
    const soiOnly = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x02]);
    expect(() => jpegDimensionsFromBuffer(soiOnly)).toThrow('No SOF marker');
  });
});
