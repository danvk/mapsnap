import type { Label } from './types';
import { LabelPreview } from './LabelPreview';

interface LabelsTableProps {
  labels: Label[];
  selectedIndex: number | null;
  showOnlyUnlabeled: boolean;
  image: HTMLImageElement | null;
  /** Preview crop size in image pixels (scaled to the image's resolution). */
  boxWidth: number;
  boxHeight: number;
  onSelect: (index: number) => void;
  onChangeText: (index: number, text: string) => void;
  onDelete: (index: number) => void;
}

/**
 * Table of labels, newest first, so a freshly added label's preview is visible
 * without scrolling. Each row shows the label's index, a cropped preview, and a
 * text box for the truth text, plus a delete button. When `showOnlyUnlabeled`
 * is set, only labels without text are shown (plus the selected one, so typing
 * into it doesn't make the row disappear). Editing flows back to the parent,
 * which persists to the labels.json sidecar.
 */
export function LabelsTable(props: LabelsTableProps) {
  const {
    labels,
    selectedIndex,
    showOnlyUnlabeled,
    image,
    boxWidth,
    boxHeight,
    onSelect,
    onChangeText,
    onDelete,
  } = props;

  const rows = labels
    .map((label, i) => ({ label, i }))
    .filter(
      ({ label, i }) =>
        !showOnlyUnlabeled || label.text.trim() === '' || i === selectedIndex,
    )
    .reverse();

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
          {rows.map(({ label, i }) => (
            <tr
              key={i}
              className={i === selectedIndex ? 'selected' : undefined}
              onClick={() => onSelect(i)}
            >
              <td>{i + 1}</td>
              <td>
                <LabelPreview
                  label={label}
                  image={image}
                  boxWidth={boxWidth}
                  boxHeight={boxHeight}
                />
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
