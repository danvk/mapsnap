/**
 * A flat-bottom dot plot: one dot per page, click to select, drag to filter.
 *
 * Deliberately not a histogram. These distributions are small (under 200
 * points) and often lumpy in ways bin edges hide -- Miami's page scales are
 * bimodal around the half- and double-scale sheet families, which a histogram
 * can smear into one mound or split into two depending on where its bins fall.
 * Every page gets its own dot at its own value; the pile's shape is the
 * distribution.
 */

import { useRef, useState } from 'react';

import { dotAt, layoutDots, scaleFor } from '../dotPlot';

/** One page's value on this metric. */
export interface DotDatum {
  /** The page's itemIndex, so a click can select it in the viewer. */
  id: number;
  value: number;
}

interface DotPlotProps {
  label: string;
  data: DotDatum[];
  /** Formats a value for the axis ends and the brush handles. */
  format: (value: number) => string;
  selectedId: number | null;
  /** The filter this chart owns, in value units, or null for none. */
  range: [number, number] | null;
  onSelect: (id: number | null) => void;
  onRangeChange: (range: [number, number] | null) => void;
  width?: number;
  height?: number;
}

const RADIUS = 3;
// Half the width of the widest label we expect, in pixels.
const LABEL_HALF_PX = 16;
const AXIS_HEIGHT = 14;
// A press that moves less than this is a click, not a drag. Small enough that
// a deliberate narrow brush still works, large enough to absorb a shaky click.
const DRAG_SLOP_PX = 3;

/** Which side to hang a label off, so it stays inside the chart at the edges. */
function edgeAnchor(x: number, width: number): 'start' | 'middle' | 'end' {
  if (x < LABEL_HALF_PX) return 'start';
  if (x > width - LABEL_HALF_PX) return 'end';
  return 'middle';
}

export function DotPlot(props: DotPlotProps) {
  const {
    label,
    data,
    format,
    selectedId,
    range,
    onSelect,
    onRangeChange,
    width = 176,
    height = 56,
  } = props;
  const svgRef = useRef<SVGSVGElement>(null);
  // While dragging: [x0, x1] in pixels. Null when not dragging.
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const pressRef = useRef<number | null>(null);

  const layout = layoutDots(
    data.map((d) => d.value),
    width,
    height,
    RADIUS,
  );
  const [low, high] = layout.domain;
  const { toX, toValue } = scaleFor(layout.domain, width, RADIUS);

  const pointerX = (event: React.PointerEvent) => {
    const box = svgRef.current?.getBoundingClientRect();
    return box ? event.clientX - box.left : 0;
  };

  const handlePointerDown = (event: React.PointerEvent) => {
    if (data.length === 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pressRef.current = pointerX(event);
    setDrag(null);
  };

  const handlePointerMove = (event: React.PointerEvent) => {
    const start = pressRef.current;
    if (start === null) return;
    const x = pointerX(event);
    if (Math.abs(x - start) >= DRAG_SLOP_PX) setDrag([start, x]);
  };

  const handlePointerUp = (event: React.PointerEvent) => {
    const start = pressRef.current;
    pressRef.current = null;
    if (start === null) return;
    const x = pointerX(event);
    if (Math.abs(x - start) < DRAG_SLOP_PX) {
      // A click: select the dot under it, or clear the filter when the click
      // lands on empty space -- which is the only way to get rid of a brush
      // without hunting for a button.
      const box = svgRef.current?.getBoundingClientRect();
      const y = box ? event.clientY - box.top : 0;
      const hit = dotAt(layout, x, y, RADIUS);
      if (hit !== null) onSelect(data[hit]?.id ?? null);
      else if (range) onRangeChange(null);
      else onSelect(null);
    } else {
      const [a, b] = [Math.min(start, x), Math.max(start, x)];
      onRangeChange([toValue(a), toValue(b)]);
    }
    setDrag(null);
  };

  const active = drag ?? (range ? [toX(range[0]), toX(range[1])] : null);
  const inRange = (value: number) =>
    !range || (value >= range[0] && value <= range[1]);

  return (
    <div className="dot-plot">
      <div className="dot-plot-label">
        <span>{label}</span>
        {range && (
          <button
            type="button"
            className="dot-plot-clear"
            onClick={() => onRangeChange(null)}
          >
            clear filter
          </button>
        )}
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={height + AXIS_HEIGHT}
        className="dot-plot-svg"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        role="img"
        aria-label={`${label}: ${data.length} pages`}
      >
        <line
          x1={0}
          y1={height + 0.5}
          x2={width}
          y2={height + 0.5}
          className="dot-plot-axis"
        />
        {active && (
          <rect
            x={Math.min(active[0], active[1])}
            y={height - 2}
            width={Math.abs(active[1] - active[0])}
            height={5}
            className="dot-plot-brush"
          />
        )}
        {layout.dots.map((dot, index) => {
          const datum = data[index];
          if (!datum) return null;
          const selected = datum.id === selectedId;
          return (
            <circle
              key={`${datum.id}-${index}`}
              cx={dot.x}
              cy={dot.y}
              r={RADIUS}
              className={
                'dot-plot-dot' +
                (inRange(datum.value) ? '' : ' is-out') +
                (selected ? ' is-selected' : '')
              }
            />
          );
        })}
        {data.length > 0 && (
          <>
            {/* The domain's ends stay put whatever the brush does: reading the
                filter bounds off the axis while the axis itself silently
                rescaled would make a narrow filter look like the whole
                distribution. */}
            <text x={0} y={height + AXIS_HEIGHT - 2} className="dot-plot-tick">
              {format(low)}
            </text>
            <text
              x={width}
              y={height + AXIS_HEIGHT - 2}
              textAnchor="end"
              className="dot-plot-tick"
            >
              {format(high)}
            </text>
            {range && (
              <>
                {/* A brush at either extreme puts its label half outside the
                    chart; anchoring it inward keeps the number readable. */}
                <text
                  x={Math.max(0, toX(range[0]))}
                  y={height - 6}
                  textAnchor={edgeAnchor(toX(range[0]), width)}
                  className="dot-plot-tick is-brush"
                >
                  {format(range[0])}
                </text>
                <text
                  x={Math.min(width, toX(range[1]))}
                  y={height - 6}
                  textAnchor={edgeAnchor(toX(range[1]), width)}
                  className="dot-plot-tick is-brush"
                >
                  {format(range[1])}
                </text>
              </>
            )}
          </>
        )}
      </svg>
    </div>
  );
}
