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

// Box color: highlight when selected, green once text is entered, else pink.
function labelColor(text: string, isSelected: boolean): string {
  if (isSelected) return '#ff6600';
  return text.trim() ? '#2e7d32' : '#d81b60';
}

/**
 * SVG overlay drawing a colored box around each label point. Boxes are colored
 * by whether their text has been entered yet, the selected label is highlighted
 * (mirroring the streets.json overlay), and any entered text is shown beside the
 * box.
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
        const color = labelColor(label.text, isSelected);
        const corners = labelBox(label.x, label.y).map(([x, y]) =>
          toDisplay(x, y),
        );
        const points = corners.map(([dx, dy]) => `${dx},${dy}`).join(' ');
        const rightX = Math.max(...corners.map(([dx]) => dx));
        const [, cy] = toDisplay(label.x, label.y);
        return (
          <g key={i}>
            <polygon
              points={points}
              fill={color}
              fillOpacity={isSelected ? 0.2 : 0.08}
              stroke={color}
              strokeWidth={isSelected ? 3 : 2}
            />
            {label.text.trim() && (
              <text
                x={rightX + 4}
                y={cy}
                fontSize={18}
                fontFamily="sans-serif"
                fontWeight="bold"
                textAnchor="start"
                dominantBaseline="middle"
                fill={color}
                stroke="white"
                strokeWidth={3}
                paintOrder="stroke"
              >
                {label.text}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
