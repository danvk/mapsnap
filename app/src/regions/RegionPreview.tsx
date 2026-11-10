import { useEffect, useMemo, useRef } from 'react';
import { previewCrop } from './polygons';
import type { RegionPolygon } from './types';

/** Largest display size of the preview in CSS pixels; a bigger crop is scaled to fit. */
export const PREVIEW_BOX: [number, number] = [320, 420];

interface RegionPreviewProps {
  /** The decoded full-resolution sheet, or null while it loads. */
  image: HTMLImageElement | null;
  /** The selected region, or null when nothing is selected. */
  region: RegionPolygon | null;
  /** The region's outline color, matching the main canvas. */
  color: string;
}

/**
 * The selected region at full resolution: the sheet around its bounding box, with
 * the ring drawn as an outline only.
 *
 * The main canvas is worked zoomed out, where a page number is a smudge, so checking
 * that a traced block really holds the number it was named after used to mean zooming
 * in and back out. This shows the crop at 1:1 (scaled down only when the block is
 * larger than the box) with no fill, so the number reads clearly.
 */
export function RegionPreview(props: RegionPreviewProps) {
  const { image, region, color } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const crop = useMemo(
    () =>
      image && region
        ? previewCrop(
            region.ring,
            [image.naturalWidth, image.naturalHeight],
            PREVIEW_BOX,
          )
        : null,
    [image, region],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image || !region || !crop) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    // Back the canvas at device resolution so 1:1 means native sharpness on a
    // high-DPI display rather than a resampled copy.
    const ratio = window.devicePixelRatio || 1;
    const displayWidth = Math.round(crop.width * crop.scale);
    const displayHeight = Math.round(crop.height * crop.scale);
    canvas.width = displayWidth * ratio;
    canvas.height = displayHeight * ratio;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.drawImage(
      image,
      crop.x,
      crop.y,
      crop.width,
      crop.height,
      0,
      0,
      displayWidth,
      displayHeight,
    );
    context.beginPath();
    region.ring.forEach(([x, y], index) => {
      const px = (x - crop.x) * crop.scale;
      const py = (y - crop.y) * crop.scale;
      if (index === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.closePath();
    context.lineWidth = 2;
    context.strokeStyle = color;
    context.stroke();
  }, [image, region, color, crop]);

  return (
    <div className="regions-preview">
      {region === null ? (
        <p className="regions-preview-empty">
          Select a region to see it at full resolution.
        </p>
      ) : crop === null ? (
        <p className="regions-preview-empty">loading the sheet…</p>
      ) : (
        <>
          <canvas ref={canvasRef} />
          <p className="regions-preview-caption">
            <span style={{ color }}>■</span>{' '}
            {region.text === '' ? 'unnamed' : `page ${region.text}`},{' '}
            {crop.width.toFixed(0)}×{crop.height.toFixed(0)} px at{' '}
            {crop.scale === 1 ? '1:1' : `${(crop.scale * 100).toFixed(0)}%`}
          </p>
        </>
      )}
    </div>
  );
}
