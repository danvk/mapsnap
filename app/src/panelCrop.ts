import type { PanelPolygon } from './types';

/** A panel's crop box in its parent's pixel frame. */
export interface PanelCrop {
  x: number;
  y: number;
  width: number;
  height: number;
  /** The panel's ring, still in the parent's frame. */
  ring: PanelPolygon;
}

/**
 * The 1-based panel index a split stem names, or null when it names no panel.
 *
 * "p20__3" -> 3, "p20" -> null. Mirrors the `__N` suffix `mapsnap split` gives
 * each panel it writes.
 */
export function panelIndexFromStem(stem: string): number | null {
  const match = /__(\d+)$/.exec(stem);
  if (!match) return null;
  const index = Number(match[1]);
  return Number.isInteger(index) && index > 0 ? index : null;
}

/** The parent stem a split stem belongs to: "p20__3" -> "p20". */
export function parentStem(stem: string): string {
  return stem.replace(/__\d+$/, '');
}

/**
 * Where `mapsnap split` cut panel `index` out of its parent.
 *
 * Mirrors write_panels: the crop is the polygon's bounding box, clamped to the
 * image, with the same asymmetric rounding -- floor on the near edges, round on
 * the far ones. Getting that wrong shifts every detection by up to a pixel,
 * which is invisible until it is not.
 */
export function panelCrop(
  panels: PanelPolygon[],
  index: number,
  width: number,
  height: number,
): PanelCrop | null {
  const ring = panels[index - 1];
  if (!ring || ring.length === 0) return null;
  const xs = ring.map(([x]) => x);
  const ys = ring.map(([, y]) => y);
  const x0 = Math.max(0, Math.trunc(Math.min(...xs)));
  const y0 = Math.max(0, Math.trunc(Math.min(...ys)));
  const x1 = Math.min(width, Math.round(Math.max(...xs)));
  const y1 = Math.min(height, Math.round(Math.max(...ys)));
  if (x1 <= x0 || y1 <= y0) return null;
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0, ring };
}

/**
 * Re-cut panel `index` from its parent image, as `mapsnap split` would have.
 *
 * The panel JPEG is cropped to the polygon's bounding box with everything
 * outside the polygon painted white, so a corpus run that keeps only the parent
 * (panel images are re-cut on the worker and never uploaded) can still be
 * viewed against the panel's own reads. Returns a canvas to draw in the image's
 * place.
 */
export function cutPanel(
  image: CanvasImageSource,
  crop: PanelCrop,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = crop.width;
  canvas.height = crop.height;
  const context = canvas.getContext('2d');
  if (!context) return canvas;
  // White first: the pixels outside the ring are white in the real panel, and
  // a non-rectangular cut leaves some of the box uncovered.
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, crop.width, crop.height);
  context.save();
  context.beginPath();
  crop.ring.forEach(([x, y], i) => {
    const px = x - crop.x;
    const py = y - crop.y;
    if (i === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  });
  context.closePath();
  context.clip();
  context.drawImage(image, -crop.x, -crop.y);
  context.restore();
  return canvas;
}
