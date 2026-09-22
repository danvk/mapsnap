/**
 * The per-page metrics the volume charts, and the filter they drive.
 *
 * Only metrics that vary meaningfully across a volume earn a chart. Skew and
 * anisotropy do not: a generated fit is a similarity, so they are 0 and 1 by
 * construction, and truth annotations stay close to that.
 */

import type { PageGeo } from './pages';

export type MetricKey = 'scale' | 'rotation';

/** An alternate axis for the same metric, offered as a checkbox on the chart. */
export interface MetricFold {
  label: string;
  apply: (value: number) => number;
}

export interface Metric {
  key: MetricKey;
  label: string;
  of: (page: PageGeo) => number;
  format: (value: number) => string;
  fold?: MetricFold;
}

export const METRICS: Metric[] = [
  {
    key: 'scale',
    label: 'Scale (px/ft)',
    of: (page) => page.scalePixelsPerFoot,
    format: (value) => value.toFixed(2),
  },
  {
    key: 'rotation',
    label: 'Rotation (deg)',
    of: (page) => page.rotationDegrees,
    format: (value) => value.toFixed(1),
    // A sheet turned a quarter turn carries the same street grid, so folding
    // onto [0, 90) lands it on top of its upright siblings -- Miami's -91.6
    // degree page joins the -1.8 degree pile at 88. The cost is that a volume
    // whose rotations straddle zero splits across the wrap, which is why both
    // axes are offered rather than one replacing the other.
    fold: { label: '0-90', apply: (deg) => ((deg % 90) + 90) % 90 },
  },
];

/**
 * A range filter on one metric, in that metric's own units.
 *
 * `folded` records which axis the range was drawn on: the same number means
 * different pages folded and unfolded, so a filter cannot be read without it.
 */
export interface MetricFilter {
  metric: MetricKey;
  folded: boolean;
  range: [number, number];
}

/** A page's value on a metric, on whichever axis is being shown. */
export function metricValue(
  metric: Metric,
  page: PageGeo,
  folded: boolean,
): number {
  const value = metric.of(page);
  return folded && metric.fold ? metric.fold.apply(value) : value;
}

/** Whether a page passes the filter; everything passes when there is none. */
export function passesFilter(
  page: PageGeo,
  filter: MetricFilter | null,
): boolean {
  if (!filter) return true;
  const metric = METRICS.find((m) => m.key === filter.metric);
  if (!metric) return true;
  const value = metricValue(metric, page, filter.folded);
  return value >= filter.range[0] && value <= filter.range[1];
}
