import { MAX_BOX_SIZE, MIN_BOX_SIZE } from './labels';

interface BoxSizeControlsProps {
  boxWidth: number;
  boxHeight: number;
  onChangeWidth: (width: number) => void;
  onChangeHeight: (height: number) => void;
}

/**
 * Sliders for the label box's size in image pixels.
 *
 * The box is purely a visualization — labels are stored as points — so these
 * only affect the overlay on the image and the crop each preview shows, and
 * nothing here is written to the sidecar.
 */
export function BoxSizeControls(props: BoxSizeControlsProps) {
  const { boxWidth, boxHeight, onChangeWidth, onChangeHeight } = props;
  const sliders = [
    { label: 'Box width', value: boxWidth, onChange: onChangeWidth },
    { label: 'Box height', value: boxHeight, onChange: onChangeHeight },
  ];
  return (
    <div className="box-size-controls">
      {sliders.map(({ label, value, onChange }) => (
        <label key={label} className="box-size-slider">
          <span className="box-size-label">{label}</span>
          <input
            type="range"
            min={MIN_BOX_SIZE}
            max={MAX_BOX_SIZE}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
          />
          <span className="box-size-value">{value}</span>
        </label>
      ))}
    </div>
  );
}
