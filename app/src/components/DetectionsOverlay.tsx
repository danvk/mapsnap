import { confidenceColor, type IndexedDetection } from '../detections';

interface DetectionsOverlayProps {
  detections: IndexedDetection[];
  selectedIndices: Set<number>;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Image size in JSON coordinate space. */
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * SVG overlay for streets mode: detection polygons over the displayed image.
 * Selected detections are highlighted and labeled with their text.
 */
export function DetectionsOverlay(props: DetectionsOverlayProps) {
  const {
    detections,
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
      {detections.map(({ det, i }) => {
        const isSelected = selectedIndices.has(i);
        const isIgnored = det.ignore === true;
        const isHint = det.hint === true;
        const color = isSelected
          ? '#ff6600'
          : isIgnored
            ? '#999'
            : isHint
              ? '#7c3aed'
              : confidenceColor(det.confidence);
        const points = det.polygon
          .map(([x, y]) => toDisplay(x, y))
          .map(([dx, dy]) => `${dx},${dy}`)
          .join(' ');
        const dashArray = isIgnored ? '4 3' : isHint ? '3 2' : undefined;

        const [lx, ly] = toDisplay(det.polygon[0][0], det.polygon[0][1]);

        return (
          <g key={i}>
            <polygon
              points={points}
              fill={color}
              fillOpacity={isSelected ? 0.25 : 0.05}
              stroke={color}
              strokeWidth={isSelected ? 2.5 : 1.2}
              strokeDasharray={dashArray}
            />
            {isSelected && (
              <text
                x={lx}
                y={ly - 4}
                fontSize={11}
                fontFamily="sans-serif"
                fill={color}
                stroke="white"
                strokeWidth={2}
                paintOrder="stroke"
              >
                {det.text}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
