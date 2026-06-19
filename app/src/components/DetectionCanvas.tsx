import { useEffect, useRef } from 'react';
import { drawDetectionCanvas } from '../detections';
import type { Detection } from '../types';

interface DetectionCanvasProps {
  det: Detection;
  image: HTMLImageElement | null;
  jsonWidth: number;
  jsonHeight: number;
}

/** A small canvas showing the rotated, deskewed image patch for one detection. */
export function DetectionCanvas(props: DetectionCanvasProps) {
  const { det, image, jsonWidth, jsonHeight } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !image) return;
    drawDetectionCanvas({
      canvas: canvasRef.current,
      det,
      image,
      jsonWidth,
      jsonHeight,
    });
  }, [det, image, jsonWidth, jsonHeight]);

  return <canvas ref={canvasRef} />;
}
