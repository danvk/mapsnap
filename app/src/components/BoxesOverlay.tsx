import { boxAngleColor, type IndexedBox } from '../boxes';

interface BoxesOverlayProps {
  /** Boxes already filtered to the visible rotations and side minimums. */
  boxes: IndexedBox[];
  selectedIndices: Set<number>;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Image size in JSON coordinate space. */
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * SVG overlay for boxes mode: the raw CRAFT detection boxes over the displayed image,
 * outlined by the rotation pass that found them. The selected box is highlighted.
 */
export function BoxesOverlay(props: BoxesOverlayProps) {
  const {
    boxes,
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
      {boxes.map(({ box, i }) => {
        const isSelected = selectedIndices.has(i);
        const color = isSelected ? '#ff6600' : boxAngleColor(box.angle);
        const points = box.polygon
          .map(([x, y]) => toDisplay(x, y))
          .map(([dx, dy]) => `${dx},${dy}`)
          .join(' ');
        return (
          <polygon
            key={i}
            points={points}
            fill={color}
            fillOpacity={isSelected ? 0.25 : 0.05}
            stroke={color}
            strokeWidth={isSelected ? 2.5 : 1.2}
          />
        );
      })}
    </svg>
  );
}
