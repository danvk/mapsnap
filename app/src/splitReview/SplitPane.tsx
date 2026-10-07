import { panelColor } from '../components/PanelsOverlay';
import type { PanelPolygon } from '../types';
import type { ArmOutcome } from './review';

interface SplitPaneProps {
  /** "A" or "B". */
  side: string;
  armLabel: string;
  outcome: ArmOutcome;
  imageUrl: string;
  width: number;
  height: number;
  panels: PanelPolygon[];
  truth: PanelPolygon[];
  showTruth: boolean;
  /** Panel overlay opacity, 0..100. */
  opacity: number;
}

// SVG points attribute for a ring.
function points(ring: PanelPolygon): string {
  return ring.map(([x, y]) => `${x},${y}`).join(' ');
}

// Label position: the ring's vertex mean, good enough for the convex-ish panels
// splits produce.
function centroid(ring: PanelPolygon): [number, number] {
  const n = Math.max(ring.length, 1);
  return [
    ring.reduce((sum, [x]) => sum + x, 0) / n,
    ring.reduce((sum, [, y]) => sum + y, 0) / n,
  ];
}

/**
 * One arm's split of the page: its panels over the image, the truth's boundaries
 * dashed on top, and how the arm scored.
 *
 * The SVG works in the image's own pixel frame (viewBox), so the panels need no
 * rescaling and the figure fits whatever space the pane has.
 */
export function SplitPane(props: SplitPaneProps) {
  const {
    side,
    armLabel,
    outcome,
    imageUrl,
    width,
    height,
    panels,
    truth,
    showTruth,
    opacity,
  } = props;
  const fontSize = Math.min(width, height) / 18;
  return (
    <div className="sr-pane">
      <div className="sr-pane-head">
        <span className="sr-arm">
          {side} · {armLabel}
        </span>
        <span className={outcome.right ? 'sr-ok' : 'sr-bad'}>
          {outcome.panels} panel{outcome.panels === 1 ? '' : 's'}
          {outcome.right ? ' ✓' : ` ✗ (truth ${outcome.truthPanels})`}
        </span>
        <span className="sr-iou">IoU {outcome.iou.toFixed(3)}</span>
      </div>
      <svg
        className="sr-figure"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <image href={imageUrl} width={width} height={height} />
        <g opacity={opacity / 100}>
          {panels.length > 1 &&
            panels.map((ring, i) => {
              const color = panelColor(i);
              const [cx, cy] = centroid(ring);
              return (
                <g key={i}>
                  <polygon
                    points={points(ring)}
                    fill={color}
                    fillOpacity={0.15}
                    stroke={color}
                    strokeWidth={3}
                    vectorEffect="non-scaling-stroke"
                  />
                  <text
                    x={cx}
                    y={cy}
                    fontSize={fontSize}
                    fontFamily="sans-serif"
                    fontWeight="bold"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill={color}
                    stroke="white"
                    strokeWidth={fontSize / 8}
                    paintOrder="stroke"
                  >
                    {i + 1}
                  </text>
                </g>
              );
            })}
        </g>
        {showTruth &&
          truth.length > 1 &&
          truth.map((ring, i) => (
            <g key={i} fill="none">
              <polygon
                points={points(ring)}
                stroke="black"
                strokeWidth={3}
                vectorEffect="non-scaling-stroke"
              />
              <polygon
                points={points(ring)}
                stroke="white"
                strokeWidth={1.5}
                strokeDasharray="6 5"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          ))}
      </svg>
    </div>
  );
}
