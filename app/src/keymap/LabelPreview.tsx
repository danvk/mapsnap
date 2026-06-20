import { useEffect, useRef } from 'react';
import { LABEL_BOX_HEIGHT, LABEL_BOX_WIDTH } from './labels';
import type { Label } from './types';

interface LabelPreviewProps {
  label: Label;
  image: HTMLImageElement | null;
}

/**
 * A thumbnail showing the image region around a label point, analogous to the
 * streets.json detection preview but without rotation. The canvas is drawn at
 * the box's pixel size; CSS scales it down for display.
 */
export function LabelPreview(props: LabelPreviewProps) {
  const { label, image } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image || !image.naturalWidth) return;
    canvas.width = LABEL_BOX_WIDTH;
    canvas.height = LABEL_BOX_HEIGHT;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crop the label's box region and draw it 1:1 into the canvas.
    ctx.drawImage(
      image,
      label.x - LABEL_BOX_WIDTH / 2,
      label.y - LABEL_BOX_HEIGHT / 2,
      LABEL_BOX_WIDTH,
      LABEL_BOX_HEIGHT,
      0,
      0,
      LABEL_BOX_WIDTH,
      LABEL_BOX_HEIGHT,
    );
  }, [label.x, label.y, image]);

  return <canvas ref={canvasRef} className="label-preview" />;
}
