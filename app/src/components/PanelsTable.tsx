import { polygonArea } from '../geometry';
import type { PanelPolygon } from '../types';

interface PanelsTableProps {
  panels: PanelPolygon[];
  selectedIndices: Set<number>;
  onSelect: (index: number) => void;
  jsonWidth: number;
  jsonHeight: number;
}

/**
 * Table of panels in reading order, showing each panel's 1-based index and its
 * area as a percentage of the full image. Clicking a row selects that panel.
 */
export function PanelsTable(props: PanelsTableProps) {
  const { panels, selectedIndices, onSelect, jsonWidth, jsonHeight } = props;
  const imageArea = jsonWidth * jsonHeight;

  return (
    <div id="detections-panel">
      <table id="detections-table">
        <thead>
          <tr>
            <th>Index</th>
            <th>Area</th>
          </tr>
        </thead>
        <tbody>
          {panels.map((panel, i) => {
            const areaPercent = imageArea
              ? (polygonArea(panel) / imageArea) * 100
              : 0;
            return (
              <tr
                key={i}
                className={selectedIndices.has(i) ? 'selected' : undefined}
                onClick={() => onSelect(i)}
              >
                <td>{i + 1}</td>
                <td>{areaPercent.toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
