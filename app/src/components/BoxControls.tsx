import { useMemo } from 'react';
import { boxAngleColor } from '../boxes';
import type { Box } from '../types';

interface BoxControlsProps {
  boxes: Box[];
  /** Rotations currently shown on the image. */
  enabledAngles: Set<number>;
  onToggleAngle: (angle: number) => void;
}

/**
 * Right-column panel for boxes mode: one checkbox per detection rotation (0/90/270°),
 * each with the group's outline color and box count, toggling that rotation's overlay.
 */
export function BoxControls(props: BoxControlsProps) {
  const { boxes, enabledAngles, onToggleAngle } = props;

  // Box count per rotation, ordered by angle for a stable checkbox list.
  const groups = useMemo(() => {
    const counts = new Map<number, number>();
    for (const box of boxes) {
      counts.set(box.angle, (counts.get(box.angle) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => a[0] - b[0]);
  }, [boxes]);

  return (
    <div id="box-controls">
      <h3>Detection boxes</h3>
      <p className="box-controls-note">
        Raw CRAFT boxes from each rotation pass, before text recognition.
      </p>
      {groups.map(([angle, count]) => (
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
  );
}
