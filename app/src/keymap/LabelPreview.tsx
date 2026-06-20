import { useEffect, useRef } from 'react';
import type { Label } from './types';

interface LabelPreviewProps {
  label: Label;
  image: HTMLImageElement | null;
  /** Crop size in image pixels (scaled to the image's resolution). */
  boxWidth: number;
  boxHeight: number;
}

/**
 * A thumbnail showing the image region around a label point, analogous to the
 * streets.json detection preview but without rotation. The canvas is drawn at
 * the box's pixel size; CSS scales it down for display.
 */
export function LabelPreview(props: LabelPreviewProps) {
  const { label, image, boxWidth, boxHeight } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image || !image.naturalWidth) return;
    canvas.width = boxWidth;
    canvas.height = boxHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crop the label's box region and draw it 1:1 into the canvas.
    ctx.drawImage(
      image,
      label.x - boxWidth / 2,
      label.y - boxHeight / 2,
      boxWidth,
      boxHeight,
      0,
      0,
      boxWidth,
      boxHeight,
    );
  }, [label.x, label.y, image, boxWidth, boxHeight]);

  return <canvas ref={canvasRef} className="label-preview" />;
}
