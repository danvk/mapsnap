import type { Label } from './types';
import { LabelPreview } from './LabelPreview';

interface LabelsTableProps {
  labels: Label[];
  selectedIndex: number | null;
  image: HTMLImageElement | null;
  onSelect: (index: number) => void;
  onChangeText: (index: number, text: string) => void;
  onDelete: (index: number) => void;
}

/**
 * Table of labels in creation order. Each row shows the label's index, a
 * cropped preview, and a text box for the truth text, plus a delete button.
 * Editing flows back to the parent, which persists to the labels.json sidecar.
 */
export function LabelsTable(props: LabelsTableProps) {
  const { labels, selectedIndex, image, onSelect, onChangeText, onDelete } =
    props;

  return (
    <div id="detections-panel">
      <table id="detections-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Preview</th>
            <th>Text</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {labels.map((label, i) => (
            <tr
              key={i}
              className={i === selectedIndex ? 'selected' : undefined}
              onClick={() => onSelect(i)}
            >
              <td>{i + 1}</td>
              <td>
                <LabelPreview label={label} image={image} />
              </td>
              <td>
                <input
                  type="text"
                  name={`label-text-${i}`}
                  className="label-text-input"
                  value={label.text}
                  placeholder="text…"
                  onFocus={() => onSelect(i)}
                  onChange={(e) => onChangeText(i, e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                />
              </td>
              <td>
                <button
                  type="button"
                  className="label-delete"
                  tabIndex={-1}
                  title="Delete label"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(i);
                  }}
                >
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
