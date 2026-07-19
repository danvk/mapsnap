import { useState, type ReactElement } from 'react';

import { rmseBucket, type PageCompareStats } from '../iiif/compare';
import type { PageGeo } from '../iiif/pages';

interface PageListProps {
  pages: PageGeo[];
  /** Per-itemIndex truth stats; null entry = truth exists but is incomparable (split). */
  stats: Map<number, PageCompareStats | null> | null;
  /** Page key → note text for the volume; keys present here get a marker. */
  notes: Map<string, string>;
  selectedItemIndex: number | null;
  onSelectPage: (itemIndex: number | null) => void;
}

type SortKey = 'page' | 'rmse' | 'rot' | 'scale' | 'rotErr' | 'scaleErr';

// Natural sort key for page keys: number, then letter suffix, then split index.
function pageSortKey(pageKey: string): [number, string, number] {
  const match = pageKey.match(/^p(\d+)([a-z]*)(?:__(\d+))?$/i);
  if (!match) return [Number.MAX_SAFE_INTEGER, pageKey, 0];
  return [Number(match[1]), match[2] ?? '', match[3] ? Number(match[3]) : 0];
}

// Color class for an RMSE value (see compare.rmseBucket for the boundaries).
function rmseClass(rmseFt: number): string {
  return `rmse-${rmseBucket(rmseFt)}`;
}

// The sortable numeric value of a column for one page, or undefined when the
// page has no value there (sorted to the end in either direction).
function sortValue(
  key: SortKey,
  page: PageGeo,
  stats: PageCompareStats | null | undefined,
): number | undefined {
  switch (key) {
    case 'page':
      return undefined; // handled by natural key comparison
    case 'rmse':
      return stats?.rmseFt;
    case 'rot':
      return page.rotationDegrees;
    case 'scale':
      return page.scalePixelsPerFoot;
    case 'rotErr':
      return stats ? Math.abs(stats.rotationErrorDegrees) : undefined;
    case 'scaleErr':
      return stats ? Math.abs(stats.scaleErrorPercent) : undefined;
  }
}

/**
 * The volume viewer's page table: one row per page in the loaded annotation,
 * with per-page stats and (when the volume has truth data) `mapsnap compare`
 * error columns. Click a header to sort — by RMSE descending by default, so
 * the worst fits are on top — and a row to select the page on the map.
 */
export function PageList(props: PageListProps) {
  const { pages, stats, notes, selectedItemIndex, onSelectPage } = props;
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({
    key: 'rmse',
    descending: true,
  });

  const hasTruth = stats !== null;
  const effectiveSort =
    !hasTruth && ['rmse', 'rotErr', 'scaleErr'].includes(sort.key)
      ? { key: 'page' as SortKey, descending: false }
      : sort;

  const sorted = [...pages].sort((a, b) => {
    const naturalOrder = comparePageKeys(a.pageKey, b.pageKey);
    if (effectiveSort.key === 'page') {
      return effectiveSort.descending ? -naturalOrder : naturalOrder;
    }
    const aValue = sortValue(effectiveSort.key, a, stats?.get(a.itemIndex));
    const bValue = sortValue(effectiveSort.key, b, stats?.get(b.itemIndex));
    if (aValue === undefined && bValue === undefined) return naturalOrder;
    if (aValue === undefined) return 1; // valueless rows always sink
    if (bValue === undefined) return -1;
    const order = aValue - bValue;
    return (effectiveSort.descending ? -order : order) || naturalOrder;
  });

  function clickHeader(key: SortKey): void {
    setSort((previous) =>
      previous.key === key
        ? { key, descending: !previous.descending }
        : { key, descending: key !== 'page' },
    );
  }

  function header(key: SortKey, label: string, title?: string): ReactElement {
    const active = effectiveSort.key === key;
    return (
      <th
        className={key === 'page' ? '' : 'numeric'}
        onClick={() => clickHeader(key)}
        title={title}
      >
        {label}
        {active ? (effectiveSort.descending ? ' ▼' : ' ▲') : ''}
      </th>
    );
  }

  return (
    <div className="page-list">
      <table>
        <thead>
          <tr>
            {header('page', 'Page')}
            {hasTruth && header('rmse', 'RMSE', 'RMSE vs truth (ft)')}
            {header('rot', 'Rot', 'Rotation from north-up (°)')}
            {header('scale', 'Scale', 'Scale (px/ft)')}
            {hasTruth &&
              header('rotErr', 'Rot Δ', 'Rotation error vs truth (°)')}
            {hasTruth &&
              header('scaleErr', 'Scale Δ', 'Scale error vs truth (%)')}
          </tr>
        </thead>
        <tbody>
          {sorted.map((page) => {
            const pageStats = stats?.get(page.itemIndex);
            return (
              <tr
                key={page.itemIndex}
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
                <td>
                  {page.pageKey}
                  {notes.has(page.pageKey) && (
                    <span
                      className="page-note-marker"
                      title={notes.get(page.pageKey)}
                    >
                      {' '}
                      📓
                    </span>
                  )}
                </td>
                {hasTruth && (
                  <td className="numeric">
                    {pageStats ? (
                      <span className={rmseClass(pageStats.rmseFt)}>
                        {pageStats.rmseFt.toFixed(0)}ft
                      </span>
                    ) : stats.has(page.itemIndex) ? (
                      <span title="split page: truth is in the parent canvas frame">
                        split
                      </span>
                    ) : (
                      <span title="no truth annotation for this page">—</span>
                    )}
                  </td>
                )}
                <td className="numeric">{page.rotationDegrees.toFixed(1)}°</td>
                <td className="numeric">
                  {page.scalePixelsPerFoot.toFixed(2)}
                </td>
                {hasTruth && (
                  <td className="numeric">
                    {pageStats
                      ? `${pageStats.rotationErrorDegrees.toFixed(2)}°`
                      : ''}
                  </td>
                )}
                {hasTruth && (
                  <td className="numeric">
                    {pageStats
                      ? `${pageStats.scaleErrorPercent.toFixed(1)}%`
                      : ''}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Natural comparison of two page keys (number, letter suffix, split index).
function comparePageKeys(a: string, b: string): number {
  const [an, as, ai] = pageSortKey(a);
  const [bn, bs, bi] = pageSortKey(b);
  return an - bn || as.localeCompare(bs) || ai - bi;
}
