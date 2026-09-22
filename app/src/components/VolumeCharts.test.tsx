import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';

import type { PageGeo } from '../iiif/pages';
import type { MetricFilter } from '../iiif/metrics';
import { VolumeCharts } from './VolumeCharts';

const noop = () => {};

/** A Fargo-like volume: one rung family plus a few quarter-turned sheets. */
const pages = [1.47, 1.48, 1.46, 2.9, 0.73, 1.47]
  .map((scale, index) => ({
    itemIndex: index,
    scalePixelsPerFoot: scale,
    rotationDegrees: index === 3 ? 90.6 : -1.5 - index * 0.1,
  }))
  .map((page) => page as unknown as PageGeo);

function render(props: Partial<Parameters<typeof VolumeCharts>[0]> = {}) {
  return renderToStaticMarkup(
    <VolumeCharts
      pages={pages}
      filter={null}
      selectedItemIndex={null}
      onFilterChange={noop}
      onSelectPage={noop}
      {...props}
    />,
  );
}

describe('VolumeCharts', () => {
  it('charts every page on every metric', () => {
    const html = render();
    expect((html.match(/class="dot-plot"/g) ?? []).length).toBe(2);
    expect((html.match(/class="dot-plot-dot/g) ?? []).length).toBe(
      2 * pages.length,
    );
  });

  it('offers the fold only where a metric has one', () => {
    // Rotation folds onto [0, 90); scale has no period to fold on, so a
    // checkbox beside it would do nothing.
    expect((render().match(/type="checkbox"/g) ?? []).length).toBe(1);
    expect(render()).toContain('0-90');
  });

  it('starts on the raw axis', () => {
    // The fold hides which pages are turned, so it is opt-in.
    expect(render()).not.toContain('checked=""');
    expect(render()).toContain('90.6');
  });

  it('marks the selected page in both charts', () => {
    const html = render({ selectedItemIndex: 3 });
    expect((html.match(/dot-plot-dot is-selected/g) ?? []).length).toBe(2);
  });

  it('shows a brush only on the metric it was drawn on', () => {
    const filter: MetricFilter = {
      metric: 'scale',
      folded: false,
      range: [1.4, 1.5],
    };
    const html = render({ filter });
    expect((html.match(/class="dot-plot-brush"/g) ?? []).length).toBe(1);
    // Two of the six pages fall outside on scale; the rotation chart, which
    // owns no filter, greys nothing.
    expect((html.match(/dot-plot-dot is-out/g) ?? []).length).toBe(2);
  });

  it('draws nothing for a volume with one page', () => {
    expect(render({ pages: pages.slice(0, 1) })).toBe('');
  });
});
