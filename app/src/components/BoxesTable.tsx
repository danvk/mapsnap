import { boxAngleColor, boxArea, type IndexedBox } from '../boxes';
import { DetectionCanvas } from './DetectionCanvas';

interface BoxesTableProps {
  boxes: IndexedBox[];
  selectedIndices: Set<number>;
  onSelect: (index: number) => void;
  image: HTMLImageElement | null;
  jsonWidth: number;
  jsonHeight: number;
}

const NUM_PREVIEW_IMAGES = 100;

/**
 * Table of detection boxes sorted by area (largest first). Shows all visible boxes, or only
 * the selected one when a selection is active. The first rows render a deskewed image patch,
 * and each row's angle is tagged with the overlay color. Clicking a row selects that box.
 */
export function BoxesTable(props: BoxesTableProps) {
  const { boxes, selectedIndices, onSelect, image, jsonWidth, jsonHeight } =
    props;

  const visible = boxes
    .filter(({ i }) => selectedIndices.size === 0 || selectedIndices.has(i))
    .sort((a, b) => boxArea(b.box) - boxArea(a.box));

  return (
    <div id="detections-panel">
      <table id="detections-table">
        <thead>
          <tr>
            <th>Angle</th>
            <th>Long</th>
            <th>Short</th>
            <th>Area</th>
            <th>Image</th>
          </tr>
        </thead>
        <tbody>
          {visible.map(({ box, i }, rowIdx) => (
            <tr
              key={i}
              className={selectedIndices.has(i) ? 'selected' : undefined}
              onClick={() => onSelect(i)}
            >
              <td>
                <span
                  className="box-swatch"
                  style={{
                    backgroundColor: boxAngleColor(box.angle),
                    marginRight: 6,
                    verticalAlign: 'middle',
                  }}
                />
                {box.angle}°
              </td>
              <td>{box.long_side}</td>
              <td>{box.short_side}</td>
              <td>{boxArea(box).toLocaleString()}</td>
              <td>
                {rowIdx < NUM_PREVIEW_IMAGES && (
                  <DetectionCanvas
                    det={box}
                    image={image}
                    jsonWidth={jsonWidth}
                    jsonHeight={jsonHeight}
                  />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
