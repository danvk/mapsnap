import { boxAngleColor } from '../boxes';
import type { Box } from '../types';

interface BoxesOverlayProps {
  boxes: Box[];
  /** Rotations whose boxes are shown; boxes at other angles are hidden. */
  enabledAngles: Set<number>;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Image size in JSON coordinate space. */
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * SVG overlay for boxes mode: the raw CRAFT detection boxes over the displayed image,
 * outlined by the rotation pass that found them and filtered to the enabled rotations.
 */
export function BoxesOverlay(props: BoxesOverlayProps) {
  const {
    boxes,
    enabledAngles,
    displayWidth,
    displayHeight,
    jsonWidth,
    jsonHeight,
  } = props;

  const toDisplay = (nx: number, ny: number): [number, number] => [
    (nx * displayWidth) / jsonWidth,
    (ny * displayHeight) / jsonHeight,
  ];

  return (
    <svg
      width={displayWidth}
      height={displayHeight}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
      }}
    >
      {boxes.map((box, i) => {
        if (!enabledAngles.has(box.angle)) return null;
        const color = boxAngleColor(box.angle);
        const points = box.polygon
          .map(([x, y]) => toDisplay(x, y))
          .map(([dx, dy]) => `${dx},${dy}`)
          .join(' ');
        return (
          <polygon
            key={i}
            points={points}
            fill={color}
            fillOpacity={0.05}
            stroke={color}
            strokeWidth={1.2}
          />
        );
      })}
    </svg>
  );
}
