import type { ReactNode } from 'react';
import type { IndexedDetection } from '../detections';
import { isOnBuildingFill } from '../detections';
import type { Detection } from '../types';
import { DetectionCanvas } from './DetectionCanvas';

interface DetectionsTableProps {
  detections: IndexedDetection[];
  selectedIndices: Set<number>;
  onSelect: (index: number) => void;
  image: HTMLImageElement | null;
  jsonWidth: number;
  jsonHeight: number;
}

const NUM_PREVIEW_IMAGES = 100;

/** Columns in the table, for the metadata row that spans them all. */
const COLUMN_COUNT = 7;

/**
 * How few rows a selection must be down to before each one expands to show where
 * it came from. Narrowing to a handful is the gesture that means "explain these".
 */
const MAX_ROWS_WITH_METADATA = 5;

/**
 * Table of detections sorted by confidence. Shows all filtered detections, or
 * only the selected ones when a selection is active. The first ten rows render
 * a deskewed image patch. Clicking a row selects that detection. Narrow the
 * selection to a few rows and each grows a metadata block explaining itself.
 */
export function DetectionsTable(props: DetectionsTableProps) {
  const {
    detections,
    selectedIndices,
    onSelect,
    image,
    jsonWidth,
    jsonHeight,
  } = props;

  const visible = detections
    .filter(({ i }) => selectedIndices.size === 0 || selectedIndices.has(i))
    .sort((a, b) => b.det.confidence - a.det.confidence);
  // Only once a selection has narrowed things down: every row of a full sheet
  // carrying a metadata block would bury the table it belongs to.
  const showMetadata =
    selectedIndices.size > 0 && visible.length <= MAX_ROWS_WITH_METADATA;

  return (
    <div id="detections-panel">
      <table id="detections-table">
        <thead>
          <tr>
            <th>Angle</th>
            <th>Long</th>
            <th>Short</th>
            <th>Conf</th>
            <th>Type</th>
            <th>Text</th>
            <th>Image</th>
          </tr>
        </thead>
        <tbody>
          {visible.map(({ det, i, relaxed }, rowIdx) => {
            const onFill = isOnBuildingFill(det);
            const classes = [
              selectedIndices.has(i) ? 'selected' : '',
              det.ignore ? 'ignored' : '',
              det.hint ? 'hint' : '',
              det.fallback ? 'fallback' : '',
              onFill ? 'on-fill' : '',
              relaxed ? 'relaxed' : '',
            ]
              .filter(Boolean)
              .join(' ');
            const type = det.ignore ? 'ignore' : det.hint ? 'hint' : 'street';
            return [
              <tr
                key={i}
                className={classes || undefined}
                onClick={() => onSelect(i)}
              >
                <td>{det.angle}</td>
                <td>{det.long_side}</td>
                <td>{det.short_side}</td>
                <td>{det.confidence.toFixed(3)}</td>
                <td>
                  {type}
                  {relaxed && (
                    <span
                      className="relaxed-badge"
                      title="Below the strict size floor; admitted because its confidence bought a lower one."
                    >
                      relaxed
                    </span>
                  )}
                  {det.underline_removed && (
                    <span
                      className="underline-badge"
                      title="An ordinal underline was painted out of this box before recognition, so this read comes from altered pixels (#250)."
                    >
                      underline
                    </span>
                  )}
                  {det.fallback && (
                    <span
                      className="fallback-badge"
                      title="Read by the key-map rectangle fallback vocabulary, not the tighter page-neighborhood radius vocabulary"
                    >
                      fallback
                    </span>
                  )}
                  {det.background && (
                    <span
                      className={
                        onFill ? 'fill-badge dropped' : 'fill-badge spared'
                      }
                      title={
                        `Background ${det.background.color} — hue ${det.background.hue}°, ` +
                        `chroma ${det.background.chroma}. ` +
                        (onFill
                          ? 'Outside the yellow/brown band, so georeferencing treats this as a ' +
                            'label on a coloured building and drops it.'
                          : 'Yellow/brown is ambiguous (aged paper, a taped-on patch, or a ' +
                            'frame building), so georeferencing keeps this.')
                      }
                    >
                      <span
                        className="fill-swatch"
                        style={{ background: det.background.color }}
                      />
                      {onFill ? 'on fill' : 'on yellow'}
                    </span>
                  )}
                </td>
                <td>{det.text}</td>
                <td>
                  {rowIdx < NUM_PREVIEW_IMAGES && (
                    <DetectionCanvas
                      det={det}
                      image={image}
                      jsonWidth={jsonWidth}
                      jsonHeight={jsonHeight}
                    />
                  )}
                </td>
              </tr>,
              showMetadata && (
                <tr key={`${i}-meta`} className="detection-meta">
                  <td colSpan={COLUMN_COUNT}>
                    <DetectionMetadata det={det} />
                  </td>
                </tr>
              ),
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Where a detection came from, for the handful of rows a small selection shows.
 *
 * Everything the row's own columns do not already say: how a key-map number was
 * repaired and on whose word (#213), the colour under a label, the flags that
 * decide whether georeferencing keeps it, and where on the sheet it sits.
 */
function DetectionMetadata({ det }: { det: Detection }) {
  const xs = det.polygon.map(([x]) => x);
  const ys = det.polygon.map(([, y]) => y);
  const center: [number, number] = [
    Math.round(xs.reduce((a, b) => a + b, 0) / xs.length),
    Math.round(ys.reduce((a, b) => a + b, 0) / ys.length),
  ];

  const rows: [string, ReactNode][] = [];
  if (det.via) rows.push(['via', det.via]);
  if (det.support !== undefined) rows.push(['support', det.support.toFixed(2)]);
  if (det.cited_by?.length) rows.push(['cited by', det.cited_by.join(', ')]);
  if (det.background) {
    rows.push([
      'background',
      <>
        <span
          className="fill-swatch"
          style={{ background: det.background.color }}
        />
        {det.background.color} — hue {det.background.hue}°, chroma{' '}
        {det.background.chroma}
      </>,
    ]);
  }
  const flags = [
    det.ignore && 'ignored',
    det.hint && 'hint',
    det.fallback && 'fallback vocabulary',
  ].filter(Boolean);
  if (flags.length) rows.push(['flags', flags.join(', ')]);
  rows.push(['center', `${center[0]}, ${center[1]} px`]);

  return (
    <dl className="detection-metadata">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
