import type { IndexedDetection } from '../detections';
import { isOnBuildingFill } from '../detections';
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

/**
 * Table of detections sorted by confidence. Shows all filtered detections, or
 * only the selected ones when a selection is active. The first ten rows render
 * a deskewed image patch. Clicking a row selects that detection.
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
          {visible.map(({ det, i }, rowIdx) => {
            const onFill = isOnBuildingFill(det);
            const classes = [
              selectedIndices.has(i) ? 'selected' : '',
              det.ignore ? 'ignored' : '',
              det.hint ? 'hint' : '',
              det.fallback ? 'fallback' : '',
              onFill ? 'on-fill' : '',
            ]
              .filter(Boolean)
              .join(' ');
            const type = det.ignore ? 'ignore' : det.hint ? 'hint' : 'street';
            return (
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
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
