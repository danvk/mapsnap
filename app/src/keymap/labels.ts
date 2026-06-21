import type { Label, LabelsJson } from './types';

/**
 * Size, in image pixels, of the box drawn around a label point and of the
 * region shown in its preview, at full image resolution. Labels are stored as
 * points; the box is only for visualization. The box is wider than tall to suit
 * horizontal page numbers.
 */
export const LABEL_BOX_WIDTH = 160;
export const LABEL_BOX_HEIGHT = 107;

/** A label box's pixel dimensions. */
export interface BoxSize {
  width: number;
  height: number;
}

/**
 * Box size scaled to the image's resolution. Full-resolution key maps (any
 * dimension over 4000px) use the full box size; smaller images are assumed to
 * be 25%-scale, so the box shrinks to match.
 */
export function labelBoxSize(imageWidth: number, imageHeight: number): BoxSize {
  const isFullScale = imageWidth > 4000 || imageHeight > 4000;
  const scale = isFullScale ? 1 : 0.25;
  return {
    width: Math.round(LABEL_BOX_WIDTH * scale),
    height: Math.round(LABEL_BOX_HEIGHT * scale),
  };
}

/** Build a labels.json payload for an image. */
export function createLabelsJson(
  image: string,
  width: number,
  height: number,
  labels: Label[],
): LabelsJson {
  return { image, width, height, labels };
}

/**
 * Axis-aligned box centered on (x, y), as a 4-point polygon in [x, y] order
 * (clockwise from top-left). Defaults to {@link LABEL_BOX_WIDTH} by
 * {@link LABEL_BOX_HEIGHT}.
 */
export function labelBox(
  x: number,
  y: number,
  width: number = LABEL_BOX_WIDTH,
  height: number = LABEL_BOX_HEIGHT,
): [number, number][] {
  const hw = width / 2;
  const hh = height / 2;
  return [
    [x - hw, y - hh],
    [x + hw, y - hh],
    [x + hw, y + hh],
    [x - hw, y + hh],
  ];
}
