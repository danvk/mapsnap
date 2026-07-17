import type { PageCompareStats } from '../iiif/compare';
import type { PageGeo } from '../iiif/pages';

interface PageListProps {
  pages: PageGeo[];
  /** Per-itemIndex truth stats; null entry = truth exists but is incomparable (split). */
  stats: Map<number, PageCompareStats | null> | null;
  selectedItemIndex: number | null;
  onSelectPage: (itemIndex: number | null) => void;
}

// Natural sort key for page keys: number, then letter suffix, then split index.
function pageSortKey(pageKey: string): [number, string, number] {
  const match = pageKey.match(/^p(\d+)([a-z]*)(?:__(\d+))?$/i);
  if (!match) return [Number.MAX_SAFE_INTEGER, pageKey, 0];
  return [Number(match[1]), match[2] ?? '', match[3] ? Number(match[3]) : 0];
}

// Color class for an RMSE value, mirroring the README's bucket boundaries.
function rmseClass(rmseFt: number): string {
  if (rmseFt <= 25) return 'rmse-good';
  if (rmseFt <= 100) return 'rmse-ok';
  return 'rmse-bad';
}

/**
 * The volume viewer's page list: one row per page in the loaded annotation,
 * with its truth RMSE when the volume has truth data. Clicking a row selects
 * the page on the map (and vice versa).
 */
export function PageList(props: PageListProps) {
  const { pages, stats, selectedItemIndex, onSelectPage } = props;
  const sorted = [...pages].sort((a, b) => {
    const [an, as, ai] = pageSortKey(a.pageKey);
    const [bn, bs, bi] = pageSortKey(b.pageKey);
    return an - bn || as.localeCompare(bs) || ai - bi;
  });
  return (
    <div className="page-list">
      <div className="page-list-header">
        <span>Page</span>
        {stats && <span>RMSE</span>}
      </div>
      <ul>
        {sorted.map((page) => {
          const pageStats = stats?.get(page.itemIndex);
          let statCell = null;
          if (stats) {
            if (pageStats) {
              statCell = (
                <span className={rmseClass(pageStats.rmseFt)}>
                  {pageStats.rmseFt.toFixed(0)}ft
                </span>
              );
            } else if (stats.has(page.itemIndex)) {
              statCell = (
                <span title="split page: truth is in the parent canvas frame">
                  split
                </span>
              );
            } else {
              statCell = (
                <span title="no truth annotation for this page">—</span>
              );
            }
          }
          return (
            <li key={page.itemIndex}>
              <button
                type="button"
                className={
                  page.itemIndex === selectedItemIndex ? 'selected' : ''
                }
                onClick={() =>
                  onSelectPage(
                    page.itemIndex === selectedItemIndex
                      ? null
                      : page.itemIndex,
                  )
                }
              >
                <span>{page.pageKey}</span>
                {statCell}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
