/**
 * JPEG dimension reading without an image library.
 *
 * Port of jpeg_dimensions in mapsnap/utils.py: scan the JPEG segment stream
 * for an SOF0–SOF3 (start of frame) marker and read the dimensions from it.
 */

import { readFileSync } from 'fs';

export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Return the dimensions of a JPEG from its bytes by scanning SOF0–SOF3 markers.
 *
 * Throws if the buffer is not a JPEG or contains no SOF marker.
 */
export function jpegDimensionsFromBuffer(data: Buffer): ImageDimensions {
  if (data.length < 2 || data[0] !== 0xff || data[1] !== 0xd8) {
    throw new Error('Not a JPEG');
  }
  let offset = 2;
  while (offset + 4 <= data.length) {
    if (data[offset] !== 0xff) break;
    const segmentType = data[offset + 1] ?? 0;
    const length = data.readUInt16BE(offset + 2);
    if (segmentType >= 0xc0 && segmentType <= 0xc3) {
      // SOF payload: precision (1 byte), then height and width (2 bytes each).
      if (offset + 9 > data.length) break;
      const height = data.readUInt16BE(offset + 5);
      const width = data.readUInt16BE(offset + 7);
      return { width, height };
    }
    offset += 2 + length;
  }
  throw new Error('No SOF marker found in JPEG');
}

/** Read a JPEG file's dimensions by scanning its SOF markers. */
export function jpegDimensions(path: string): ImageDimensions {
  return jpegDimensionsFromBuffer(readFileSync(path));
}
