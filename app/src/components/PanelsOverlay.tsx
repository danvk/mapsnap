import type { PanelPolygon } from '../types';

interface PanelsOverlayProps {
  panels: PanelPolygon[];
  /** Optional per-panel display label (e.g. page number); defaults to the 1-based index. */
  labels?: string[];
  selectedIndices: Set<number>;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Image size in JSON coordinate space. */
  jsonWidth: number;
  jsonHeight: number;
}

// A distinct, stable color for each panel index.
export function panelColor(index: number): string {
  return `hsl(${(index * 67) % 360}, 70%, 45%)`;
}

/**
 * SVG overlay for panels mode: the panel polygons over the displayed image,
 * each labeled with its 1-based index. The selected panel is highlighted.
 */
export function PanelsOverlay(props: PanelsOverlayProps) {
  const {
    panels,
    labels,
    selectedIndices,
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
      {panels.map((panel, i) => {
        const isSelected = selectedIndices.has(i);
        const color = isSelected ? '#ff6600' : panelColor(i);
        const display = panel.map(([x, y]) => toDisplay(x, y));
        const points = display.map(([dx, dy]) => `${dx},${dy}`).join(' ');
        const cx = display.reduce((s, [dx]) => s + dx, 0) / display.length;
        const cy = display.reduce((s, [, dy]) => s + dy, 0) / display.length;

        return (
          <g key={i}>
            <polygon
              points={points}
              fill={color}
              fillOpacity={isSelected ? 0.3 : 0.12}
              stroke={color}
              strokeWidth={isSelected ? 3 : 2}
            />
            <text
              x={cx}
              y={cy}
              fontSize={24}
              fontFamily="sans-serif"
              fontWeight="bold"
              textAnchor="middle"
              dominantBaseline="middle"
              fill={color}
              stroke="white"
              strokeWidth={3}
              paintOrder="stroke"
            >
              {labels?.[i] ?? i + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
