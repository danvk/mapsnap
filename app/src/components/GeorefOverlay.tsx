import type { IntersectionPoint, Street } from '../types';

interface GeorefOverlayProps {
  streets: Street[];
  intersections: IntersectionPoint[];
  showStreetsOnImage: boolean;
  showIntersectionsOnImage: boolean;
  colorByInlier: boolean;
  /** Rendered image size in CSS pixels. */
  displayWidth: number;
  displayHeight: number;
  /** Image size in JSON coordinate space. */
  jsonWidth: number;
  jsonHeight: number;
}

// Priority for stacking intersection markers (higher = drawn on top).
function intersectionPriority(ix: IntersectionPoint): number {
  return ix.initial ? 2 : ix.inlier ? 1 : 0;
}

/**
 * SVG overlay for georef mode: street label dots with direction arrows and
 * intersection GCP markers, positioned over the displayed image.
 */
export function GeorefOverlay(props: GeorefOverlayProps) {
  const {
    streets,
    intersections,
    showStreetsOnImage,
    showIntersectionsOnImage,
    colorByInlier,
    displayWidth,
    displayHeight,
    jsonWidth,
    jsonHeight,
  } = props;

  // Scale JSON coordinate space to SVG display coords.
  const toDisplay = (nx: number, ny: number): [number, number] => [
    (nx * displayWidth) / jsonWidth,
    (ny * displayHeight) / jsonHeight,
  ];

  // Render outliers first so inliers appear on top when they overlap.
  const sortedStreets = [...streets].sort(
    (a, b) => (a.inlier ? 1 : 0) - (b.inlier ? 1 : 0),
  );

  // Render in ascending priority so initial seeds appear on top.
  const sortedIntersections = [...intersections].sort(
    (a, b) => intersectionPriority(a) - intersectionPriority(b),
  );

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
      {showStreetsOnImage &&
        sortedStreets.map((st, idx) => {
          const [cx, cy] = toDisplay(st.x, st.y);
          const color = !colorByInlier
            ? '#ff0000'
            : st.inlier !== false
              ? 'orange'
              : '#888888';

          let arrows: React.ReactNode = null;
          if (st.dir_x !== undefined && st.dir_y !== undefined) {
            const arrowLen = 40;
            const dx = (st.dir_x * displayWidth) / jsonWidth;
            const dy = (st.dir_y * displayHeight) / jsonHeight;
            const len = Math.sqrt(dx * dx + dy * dy);
            const ndx = (dx / len) * arrowLen;
            const ndy = (dy / len) * arrowLen;
            arrows = ([1, -1] as const).map((sign) => (
              <line
                key={sign}
                x1={cx}
                y1={cy}
                x2={cx + sign * ndx}
                y2={cy + sign * ndy}
                stroke={color}
                strokeWidth={2}
                strokeOpacity={0.9}
              />
            ));
          }

          return (
            <g key={idx}>
              <circle
                cx={cx}
                cy={cy}
                r={8}
                fill={color}
                fillOpacity={0.7}
                stroke="white"
                strokeWidth={2}
              />
              {arrows}
              <text
                x={cx + 12}
                y={cy - 4}
                fontSize={13}
                fontFamily="sans-serif"
                fontWeight="bold"
                fill={color}
                stroke="white"
                strokeWidth={3}
                paintOrder="stroke"
              >
                {st.street}
              </text>
            </g>
          );
        })}

      {showIntersectionsOnImage &&
        sortedIntersections.map((ix, idx) => {
          const [cx, cy] = toDisplay(ix.x, ix.y);
          const color = ix.initial
            ? '#0080ff'
            : ix.inlier
              ? '#ff0000'
              : '#e6b800';
          return (
            <g key={idx}>
              <circle
                cx={cx}
                cy={cy}
                r={6}
                fill={color}
                fillOpacity={0.85}
                stroke="white"
                strokeWidth={2}
              />
              <text
                x={cx + 10}
                y={cy - 2}
                fontSize={10}
                fontFamily="sans-serif"
                fill={color}
                stroke="white"
                strokeWidth={2}
                paintOrder="stroke"
              >
                {[ix.label_a, ix.label_b].map((name, i) => (
                  <tspan key={i} x={cx + 10} dy={i === 0 ? '0' : '12'}>
                    {name}
                  </tspan>
                ))}
              </text>
            </g>
          );
        })}
    </svg>
  );
}
