import type { IndexedDetection } from '../detections';
import { DetectionCanvas } from './DetectionCanvas';

interface DetectionsTableProps {
  detections: IndexedDetection[];
  selectedIndices: Set<number>;
  onSelect: (index: number) => void;
  image: HTMLImageElement | null;
  jsonWidth: number;
  jsonHeight: number;
}

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
            const classes = [
              selectedIndices.has(i) ? 'selected' : '',
              det.ignore ? 'ignored' : '',
              det.hint ? 'hint' : '',
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
                <td>{type}</td>
                <td>{det.text}</td>
                <td>
                  {rowIdx < 10 && (
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
