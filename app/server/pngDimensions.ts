/**
 * PNG dimension reading without an image library.
 *
 * A PNG's first chunk is always IHDR, so the size sits at fixed offsets: the
 * 8-byte signature, a 4-byte chunk length, the 4-byte type "IHDR", then width
 * and height as big-endian 32-bit integers.
 */

import { readFileSync } from 'fs';
import type { ImageDimensions } from './jpegDimensions.ts';

const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/**
 * Return the dimensions of a PNG from its bytes.
 *
 * Throws if the buffer is not a PNG or its first chunk is not IHDR.
 */
export function pngDimensionsFromBuffer(data: Buffer): ImageDimensions {
  if (data.length < 24 || !data.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error('Not a PNG');
  }
  if (data.toString('latin1', 12, 16) !== 'IHDR') {
    throw new Error('PNG does not start with an IHDR chunk');
  }
  return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
}

/** Read a PNG file's dimensions from its IHDR chunk. */
export function pngDimensions(path: string): ImageDimensions {
  return pngDimensionsFromBuffer(readFileSync(path));
}
