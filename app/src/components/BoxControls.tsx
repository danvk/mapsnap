import { boxAngleColor } from '../boxes';
import type { DetectionFilters } from '../detections';

interface BoxControlsProps {
  /** Present rotations paired with their (side-filtered) box counts, ordered by angle. */
  angleGroups: [number, number][];
  enabledAngles: Set<number>;
  onToggleAngle: (angle: number) => void;
  filters: DetectionFilters;
  setFilters: (filters: DetectionFilters) => void;
}

/**
 * Image-column controls for boxes mode: one checkbox per detection rotation (0/90/270°),
 * each with the group's overlay color and current box count, plus the short/long-side
 * minimum sliders carried over from streets mode (confidence does not apply to raw boxes).
 */
export function BoxControls(props: BoxControlsProps) {
  const { angleGroups, enabledAngles, onToggleAngle, filters, setFilters } =
    props;

  return (
    <>
      <p className="box-controls-note">
        Raw CRAFT boxes from each rotation pass, before text recognition.
      </p>
      <div id="box-angle-toggles">
        {angleGroups.map(([angle, count]) => (
          <label key={angle} className="box-angle-row">
            <input
              type="checkbox"
              checked={enabledAngles.has(angle)}
              onChange={() => onToggleAngle(angle)}
            />
            <span
              className="box-swatch"
              style={{ backgroundColor: boxAngleColor(angle) }}
            />
            <span className="box-angle-label">{angle}°</span>
            <span className="box-angle-count">{count}</span>
          </label>
        ))}
      </div>
      <div id="detection-filters">
        <div className="filter-row">
          <label htmlFor="box-filter-short">
            Min short side: <span>{filters.minShortSide.toFixed(0)}</span>
          </label>
          <input
            type="range"
            id="box-filter-short"
            min={0}
            max={200}
            step={1}
            value={filters.minShortSide}
            onChange={(e) =>
              setFilters({
                ...filters,
                minShortSide: parseFloat(e.target.value),
              })
            }
          />
        </div>
        <div className="filter-row">
          <label htmlFor="box-filter-long">
            Min long side: <span>{filters.minLongSide.toFixed(0)}</span>
          </label>
          <input
            type="range"
            id="box-filter-long"
            min={0}
            max={200}
            step={1}
            value={filters.minLongSide}
            onChange={(e) =>
              setFilters({
                ...filters,
                minLongSide: parseFloat(e.target.value),
              })
            }
          />
        </div>
      </div>
    </>
  );
}
