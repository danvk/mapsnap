import { sheetFraction } from './polygons';
import type { RegionPolygon } from './types';

interface RegionsTableProps {
  regions: RegionPolygon[];
  width: number;
  height: number;
  selectedIndex: number | null;
  /** Page keys used by more than one region, which is always a mistake. */
  duplicates: ReadonlySet<string>;
  onSelect: (index: number) => void;
  onSetText: (index: number, text: string) => void;
  onDelete: (index: number) => void;
}

/**
 * One row per traced region: its page key, how much of the sheet it covers, and how
 * many vertices it has.
 *
 * The sheet-share column is here because it is the quantity that made a leaked
 * segmentation catastrophic — a hand-traced sheet is also the reference for judging
 * what a plausible share looks like.
 */
export function RegionsTable(props: RegionsTableProps) {
  const {
    regions,
    width,
    height,
    selectedIndex,
    duplicates,
    onSelect,
    onSetText,
    onDelete,
  } = props;

  return (
    <div className="regions-panel">
      <table className="regions-table">
        <thead>
          <tr>
            <th>Page</th>
            <th className="numeric">% sheet</th>
            <th className="numeric">pts</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {regions.map((region, index) => {
            const duplicate = region.text !== '' && duplicates.has(region.text);
            return (
              <tr
                key={index}
                className={
                  [
                    index === selectedIndex ? 'selected' : '',
                    region.text === '' ? 'unlabeled' : '',
                    duplicate ? 'duplicate' : '',
                  ]
                    .filter(Boolean)
                    .join(' ') || undefined
                }
                onClick={() => onSelect(index)}
              >
                <td>
                  <input
                    value={region.text}
                    placeholder="page"
                    size={6}
                    title={
                      duplicate ? 'another region claims this page' : undefined
                    }
                    onChange={(e) => onSetText(index, e.target.value)}
                    onFocus={() => onSelect(index)}
                  />
                </td>
                <td className="numeric">
                  {(sheetFraction(region.ring, width, height) * 100).toFixed(2)}
                </td>
                <td className="numeric">{region.ring.length}</td>
                <td>
                  <button
                    type="button"
                    title="Delete this region"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(index);
                    }}
                  >
                    ×
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
