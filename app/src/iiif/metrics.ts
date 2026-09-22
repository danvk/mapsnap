/**
 * The per-page metrics the volume summary charts, and the filter they drive.
 *
 * Only metrics that vary meaningfully across a volume earn a chart. Skew and
 * anisotropy do not: a generated fit is a similarity, so they are 0 and 1 by
 * construction, and truth annotations stay close to that.
 */

import type { PageGeo } from './pages';

export type MetricKey = 'scale' | 'rotation';

export interface Metric {
  key: MetricKey;
  label: string;
  of: (page: PageGeo) => number;
  format: (value: number) => string;
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
  },
];

/** A range filter on one metric, in that metric's own units. */
export interface MetricFilter {
  metric: MetricKey;
  range: [number, number];
}

/** Whether a page passes the filter; everything passes when there is none. */
export function passesFilter(
  page: PageGeo,
  filter: MetricFilter | null,
): boolean {
  if (!filter) return true;
  const metric = METRICS.find((m) => m.key === filter.metric);
  if (!metric) return true;
  const value = metric.of(page);
  return value >= filter.range[0] && value <= filter.range[1];
}
