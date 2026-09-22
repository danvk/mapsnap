import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';

import { DotPlot } from './DotPlot';

const noop = () => {};

/** Miami's page scales: bimodal around the half- and double-scale families. */
const data = [2.74, 2.81, 2.9, 5.5, 5.62, 5.71, 6.12].map((value, id) => ({
  id,
  value,
}));

function render(props: Partial<Parameters<typeof DotPlot>[0]> = {}) {
  return renderToStaticMarkup(
    <DotPlot
      label="Scale (px/ft)"
      data={data}
      format={(v) => v.toFixed(2)}
      selectedId={null}
      range={null}
      onSelect={noop}
      onRangeChange={noop}
      {...props}
    />,
  );
}

describe('DotPlot', () => {
  it('draws one dot per page, and the range ends as ticks', () => {
    const html = render();
    expect((html.match(/class="dot-plot-dot/g) ?? []).length).toBe(data.length);
    expect(html).toContain('2.74');
    expect(html).toContain('6.12');
  });

  it('greys the dots a filter excludes rather than dropping them', () => {
    const html = render({ range: [2.7, 3.0] });
    // Four of the seven fall outside, and are still drawn.
    expect((html.match(/dot-plot-dot is-out/g) ?? []).length).toBe(4);
    expect((html.match(/class="dot-plot-dot/g) ?? []).length).toBe(data.length);
  });

  it('shows the filter bounds without moving the axis ends', () => {
    // Replacing the end labels with the brush bounds made a narrow filter look
    // like the whole distribution -- the dots outside it are still drawn, so
    // the axis has to keep saying where the data really ends.
    const html = render({ range: [2.7, 3.0] });
    expect(html).toContain('2.70');
    expect(html).toContain('3.00');
    expect(html).toContain('2.74');
    expect(html).toContain('6.12');
    expect(html).toContain('clear filter');
  });

  it('marks the selected page', () => {
    expect(render({ selectedId: 3 })).toContain('dot-plot-dot is-selected');
    expect(render()).not.toContain('is-selected');
  });

  it('draws the selected dot last, so nothing covers it', () => {
    // SVG has no z-index. Selecting the first page -- the one that would
    // otherwise be painted under every dot after it -- has to move it to the
    // end of the markup, or a deep pile hides it.
    const html = render({ selectedId: 0 });
    const selected = html.indexOf('is-selected');
    const lastPlain = html.lastIndexOf('class="dot-plot-dot"');
    expect(selected).toBeGreaterThan(lastPlain);
  });

  it('keeps the selected dot opaque even when a filter excludes it', () => {
    // is-out would otherwise grey it out and drop it to half opacity, which is
    // the opposite of standing out.
    const html = render({ selectedId: 0, range: [5.5, 6.2] });
    expect(html).toContain('dot-plot-dot is-out is-selected');
  });

  it('renders nothing to click for an empty volume', () => {
    const html = render({ data: [] });
    expect(html).not.toContain('dot-plot-dot');
  });
});

describe('brush labels at the edges', () => {
  it('hangs a label inward rather than off the chart', () => {
    // A brush covering the whole range puts both labels on the boundary.
    const html = render({ range: [2.74, 6.12] });
    expect(html).toContain('text-anchor="start"');
    expect(html).toContain('text-anchor="end"');
  });
});
