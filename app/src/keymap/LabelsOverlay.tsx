import type { Label } from './types';
import { labelBox } from './labels';

interface LabelsOverlayProps {
  labels: Label[];
  selectedIndex: number | null;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Natural image size in pixels. */
  imageWidth: number;
  imageHeight: number;
}

/**
 * SVG overlay drawing a colored box around each label point, with its 1-based
 * index. The selected label is highlighted, mirroring the streets.json overlay.
 */
export function LabelsOverlay(props: LabelsOverlayProps) {
  const {
    labels,
    selectedIndex,
    displayWidth,
    displayHeight,
    imageWidth,
    imageHeight,
  } = props;

  const toDisplay = (nx: number, ny: number): [number, number] => [
    (nx * displayWidth) / imageWidth,
    (ny * displayHeight) / imageHeight,
  ];

  return (
    <svg
      width={displayWidth}
      height={displayHeight}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {labels.map((label, i) => {
        const isSelected = i === selectedIndex;
        const color = isSelected ? '#ff6600' : '#1e88e5';
        const points = labelBox(label.x, label.y)
          .map(([x, y]) => toDisplay(x, y))
          .map(([dx, dy]) => `${dx},${dy}`)
          .join(' ');
        const [cx, cy] = toDisplay(label.x, label.y);
        return (
          <g key={i}>
            <polygon
              points={points}
              fill={color}
              fillOpacity={isSelected ? 0.2 : 0.08}
              stroke={color}
              strokeWidth={isSelected ? 3 : 2}
            />
            <text
              x={cx}
              y={cy}
              fontSize={18}
              fontFamily="sans-serif"
              fontWeight="bold"
              textAnchor="middle"
              dominantBaseline="middle"
              fill={color}
              stroke="white"
              strokeWidth={3}
              paintOrder="stroke"
            >
              {i + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
