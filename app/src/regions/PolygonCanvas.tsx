import { useEffect, useRef, useState } from 'react';
import {
  MIN_RECT_DRAG,
  MIN_RING_VERTICES,
  VERTEX_HIT_RADIUS,
  rectRing,
  regionAt,
  sheetFraction,
  vertexAt,
} from './polygons';
import type { DrawMode, KeymapDetection, RegionPolygon } from './types';

interface PolygonCanvasProps {
  imageSrc: string;
  width: number;
  height: number;
  /** Displayed size = image size x zoom; the container scrolls to pan. */
  zoom: number;
  /** 'polygon' clicks out vertices; 'rectangle' drags out two corners. */
  mode: DrawMode;
  regions: RegionPolygon[];
  /** Ring being drawn, or [] when not drawing. */
  draft: [number, number][];
  selectedIndex: number | null;
  detections: readonly KeymapDetection[];
  showDetections: boolean;
  onAddVertex: (point: [number, number]) => void;
  /** Replace the draft outright, as a rectangle drag does on every move. */
  onSetDraft: (ring: [number, number][]) => void;
  /**
   * Finish a region. A rectangle drag passes its ring directly rather than relying on
   * the draft it just set: a quick drag can put the pointerup in the same tick as the
   * last pointermove, and React will not have flushed that state yet.
   */
  onCloseDraft: (ring?: [number, number][]) => void;
  onSelect: (index: number | null) => void;
  onMoveVertex: (region: number, vertex: number, to: [number, number]) => void;
}

/** Colour cycle for finished regions; index-based so a region keeps its colour. */
const COLORS = [
  '#e6194b',
  '#3cb44b',
  '#4363d8',
  '#f58231',
  '#911eb4',
  '#008080',
  '#9a6324',
  '#800000',
];

/**
 * Shading laid over every traced region, so covered ground reads at a glance.
 *
 * One neutral tint for all of them rather than each region's own colour: a key map
 * is already a field of saturated pastels, and a faint wash of eight different hues
 * disappears into it, which is exactly what a per-region fill did. A single dark
 * tint instead mutes what is done and leaves untraced blocks at full brightness.
 * Each region keeps its colour in its outline, and the selected one still fills
 * with its own colour so there is no doubt which is being edited.
 */
const TRACED_TINT = '#10243f';
const TRACED_TINT_OPACITY = 0.25;
const SELECTED_FILL_OPACITY = 0.3;

function ringPoints(ring: [number, number][], zoom: number): string {
  return ring.map(([x, y]) => `${x * zoom},${y * zoom}`).join(' ');
}

/**
 * The sheet with its regions drawn over it, and the pointer interactions for
 * tracing new ones.
 *
 * Zoom is a plain CSS-size multiplier on both the image and the SVG overlay, so the
 * browser's own scrolling does the panning — a key map is ~5600x8300 px and needs to
 * be worked at 1:1, which no fit-to-window view can offer.
 *
 * Drawing: click to drop vertices; click the first vertex again (or double-click) to
 * close. Clicking a finished region selects it; dragging one of its vertices moves
 * that vertex. Hit radii are in screen pixels, so targets keep their size at any zoom.
 */
export function PolygonCanvas(props: PolygonCanvasProps) {
  const {
    imageSrc,
    width,
    height,
    zoom,
    mode,
    regions,
    draft,
    selectedIndex,
    detections,
    showDetections,
    onAddVertex,
    onSetDraft,
    onCloseDraft,
    onSelect,
    onMoveVertex,
  } = props;

  const wrapperRef = useRef<HTMLDivElement>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const dragRef = useRef<{ region: number; vertex: number } | null>(null);
  // Anchor corner of a rectangle drag in progress, in image pixels.
  const rectAnchorRef = useRef<[number, number] | null>(null);

  // Image-pixel position of a pointer event.
  function imagePoint(
    event: React.PointerEvent | React.MouseEvent,
  ): [number, number] {
    const rect = wrapperRef.current!.getBoundingClientRect();
    return [
      (event.clientX - rect.left) / zoom,
      (event.clientY - rect.top) / zoom,
    ];
  }

  function handlePointerDown(event: React.PointerEvent): void {
    if (event.button !== 0) return;
    const [x, y] = imagePoint(event);
    const radius = VERTEX_HIT_RADIUS / zoom;

    if (draft.length > 0) {
      // Returning to the first vertex closes the ring.
      const first = draft[0]!;
      if (
        draft.length >= MIN_RING_VERTICES &&
        Math.hypot(first[0] - x, first[1] - y) <= radius
      ) {
        onCloseDraft();
        return;
      }
      onAddVertex([x, y]);
      return;
    }

    // Not drawing. A handle of the selected region starts a drag; a finished region
    // gets selected; empty sheet starts a new ring — which is the only way to begin
    // one, so it must not be shadowed by the two cases above.
    if (selectedIndex !== null) {
      const vertex = vertexAt(regions[selectedIndex]!, x, y, radius);
      if (vertex !== null) {
        dragRef.current = { region: selectedIndex, vertex };
        (event.target as Element).setPointerCapture?.(event.pointerId);
        return;
      }
    }
    const hit = regionAt(regions, x, y);
    if (hit !== null) {
      onSelect(hit);
      return;
    }
    onSelect(null);
    if (mode === 'rectangle') {
      // Anchor a drag rather than dropping a vertex; the ring is built on move.
      rectAnchorRef.current = [x, y];
      (event.target as Element).setPointerCapture?.(event.pointerId);
      return;
    }
    onAddVertex([x, y]);
  }

  function handlePointerMove(event: React.PointerEvent): void {
    const point = imagePoint(event);
    setCursor(point);
    const drag = dragRef.current;
    if (drag) {
      onMoveVertex(drag.region, drag.vertex, point);
      return;
    }
    const anchor = rectAnchorRef.current;
    if (anchor) onSetDraft(rectRing(anchor, point));
  }

  function handlePointerUp(event: React.PointerEvent): void {
    dragRef.current = null;
    const anchor = rectAnchorRef.current;
    if (!anchor) return;
    rectAnchorRef.current = null;
    const [x, y] = imagePoint(event);
    const minimum = MIN_RECT_DRAG / zoom;
    if (
      Math.abs(x - anchor[0]) < minimum ||
      Math.abs(y - anchor[1]) < minimum
    ) {
      onSetDraft([]); // a click, not a drag
      return;
    }
    onCloseDraft(rectRing(anchor, [x, y]));
  }

  // Double-click closes the ring too, which is the habit most drawing tools train.
  function handleDoubleClick(event: React.MouseEvent): void {
    if (draft.length >= MIN_RING_VERTICES) {
      event.preventDefault();
      onCloseDraft();
    }
  }

  useEffect(() => {
    if (draft.length === 0) setCursor(null);
  }, [draft.length]);

  const draftFraction = sheetFraction(draft, width, height);

  return (
    <div className="polygon-scroll">
      <div
        ref={wrapperRef}
        className="polygon-wrapper"
        style={{ width: width * zoom, height: height * zoom }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={handleDoubleClick}
      >
        <img
          src={imageSrc}
          width={width * zoom}
          height={height * zoom}
          draggable={false}
        />
        <svg
          className="polygon-overlay"
          width={width * zoom}
          height={height * zoom}
        >
          {regions.map((region, index) => {
            const color = COLORS[index % COLORS.length]!;
            const selected = index === selectedIndex;
            return (
              <g key={index}>
                <polygon
                  points={ringPoints(region.ring, zoom)}
                  fill={selected ? color : TRACED_TINT}
                  fillOpacity={
                    selected ? SELECTED_FILL_OPACITY : TRACED_TINT_OPACITY
                  }
                  stroke={color}
                  strokeWidth={selected ? 4 : 3}
                />
                {selected &&
                  region.ring.map(([x, y], vertex) => (
                    <circle
                      key={vertex}
                      cx={x * zoom}
                      cy={y * zoom}
                      r={VERTEX_HIT_RADIUS / 2}
                      fill="#fff"
                      stroke={color}
                      strokeWidth={2}
                    />
                  ))}
                {region.text !== '' && region.ring.length > 0 && (
                  <text
                    className="polygon-label"
                    x={
                      (region.ring.reduce((s, p) => s + p[0], 0) /
                        region.ring.length) *
                      zoom
                    }
                    y={
                      (region.ring.reduce((s, p) => s + p[1], 0) /
                        region.ring.length) *
                      zoom
                    }
                  >
                    {region.text}
                  </text>
                )}
              </g>
            );
          })}

          {showDetections &&
            detections.map((detection, index) => (
              <g key={`d${index}`}>
                <circle
                  cx={detection.x * zoom}
                  cy={detection.y * zoom}
                  r={3}
                  fill="#111"
                  fillOpacity={0.7}
                />
                <text
                  className="polygon-detection"
                  x={detection.x * zoom + 5}
                  y={detection.y * zoom - 5}
                >
                  {detection.text}
                </text>
              </g>
            ))}

          {draft.length > 0 && (
            <>
              {/* A rectangle drag already holds every corner, so the rubber-band to
                  the cursor belongs only to the click-out-vertices mode. */}
              <polyline
                points={ringPoints(
                  mode === 'polygon' && cursor ? [...draft, cursor] : draft,
                  zoom,
                )}
                fill="#ffd400"
                fillOpacity={0.2}
                stroke="#ffd400"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
              {mode === 'polygon' &&
                draft.map(([x, y], index) => (
                  <circle
                    key={index}
                    cx={x * zoom}
                    cy={y * zoom}
                    r={index === 0 ? VERTEX_HIT_RADIUS / 2 + 1 : 3}
                    fill={index === 0 ? '#ffd400' : '#fff'}
                    stroke="#8a6d00"
                    strokeWidth={2}
                  />
                ))}
            </>
          )}
        </svg>
        {draft.length >= MIN_RING_VERTICES && (
          <div
            className="polygon-hint"
            style={{
              left: (draft[0]![0] * zoom).toFixed(0) + 'px',
              top: (draft[0]![1] * zoom - 26).toFixed(0) + 'px',
            }}
          >
            {(draftFraction * 100).toFixed(1)}% of sheet
          </div>
        )}
      </div>
    </div>
  );
}
