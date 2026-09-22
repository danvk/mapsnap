/**
 * The volume's metric charts, in their own panel below the page info.
 *
 * A fixed home of their own, rather than appended to the info panel: that panel
 * swaps between a volume summary and a page's stats, and charts riding along
 * with it jumped down the screen every time the selection changed.
 */

import { useState, type Dispatch, type SetStateAction } from 'react';

import {
  METRICS,
  metricValue,
  type MetricFilter,
  type MetricKey,
} from '../iiif/metrics';
import type { PageGeo } from '../iiif/pages';
import { DotPlot } from './DotPlot';

interface VolumeChartsProps {
  /** Every georeferenced page, never the filtered subset. */
  pages: PageGeo[];
  filter: MetricFilter | null;
  /** itemIndex of the selected page, marked in each chart. */
  selectedItemIndex: number | null;
  /**
   * The filter setter itself, not a plain callback: toggling a fold has to
   * clear a filter it invalidates, and reading the current filter out of this
   * render's closure to decide would go stale between the brush and the toggle.
   */
  onFilterChange: Dispatch<SetStateAction<MetricFilter | null>>;
  onSelectPage: (itemIndex: number | null) => void;
}

export function VolumeCharts(props: VolumeChartsProps) {
  const { pages, filter, selectedItemIndex, onFilterChange, onSelectPage } =
    props;
  // Which metrics are being shown on their folded axis.
  const [folded, setFolded] = useState<Set<MetricKey>>(new Set());

  // One dot is not a distribution, and the axis would have no range to label.
  if (pages.length < 2) return null;

  const toggleFold = (key: MetricKey) => {
    setFolded((current) => {
      const next = new Set(current);
      if (!next.delete(key)) next.add(key);
      return next;
    });
    // A range drawn on one axis means different pages on the other, so it
    // cannot survive the switch. Dropping it beats keeping a filter whose
    // bounds no longer match anything on screen.
    onFilterChange((current) => (current?.metric === key ? null : current));
  };

  return (
    <div className="volume-charts">
      {METRICS.map((metric) => {
        const isFolded = folded.has(metric.key);
        return (
          <div key={metric.key} className="volume-chart">
            <DotPlot
              label={metric.label}
              data={pages.map((page) => ({
                id: page.itemIndex,
                value: metricValue(metric, page, isFolded),
              }))}
              format={metric.format}
              selectedId={selectedItemIndex}
              range={filter?.metric === metric.key ? filter.range : null}
              onSelect={onSelectPage}
              onRangeChange={(range) =>
                onFilterChange(
                  range
                    ? { metric: metric.key, folded: isFolded, range }
                    : null,
                )
              }
            />
            {metric.fold && (
              <label className="volume-chart-fold">
                <input
                  type="checkbox"
                  checked={isFolded}
                  onChange={() => toggleFold(metric.key)}
                />
                {metric.fold.label}
              </label>
            )}
          </div>
        );
      })}
    </div>
  );
}
