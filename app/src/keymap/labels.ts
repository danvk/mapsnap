import type { Label, LabelsWriteRequest } from './types';

/**
 * Starting size, in image pixels, of the box drawn around a label point and of
 * the region shown in its preview. Labels are stored as points and the box is
 * only for visualization, so it never reaches the sidecar — the labeler's
 * sliders adjust it live, starting wider than tall to suit horizontal page
 * numbers.
 */
export const DEFAULT_BOX_WIDTH = 160;
export const DEFAULT_BOX_HEIGHT = 107;

/** Range the box-size sliders span, in image pixels. */
export const MIN_BOX_SIZE = 20;
export const MAX_BOX_SIZE = 400;

/**
 * Build the labels.json payload for a page. The `image` field is left to the
 * server, which resolves the page's stem to an actual image file.
 */
export function createLabelsJson(
  width: number,
  height: number,
  labels: Label[],
): LabelsWriteRequest {
  return { width, height, labels };
}

/**
 * Axis-aligned box of the given size centered on (x, y), as a 4-point polygon
 * in [x, y] order (clockwise from top-left).
 */
export function labelBox(
  x: number,
  y: number,
  width: number,
  height: number,
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
