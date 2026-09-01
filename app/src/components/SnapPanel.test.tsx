import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { SnapPanel } from './SnapPanel';
import type { SnapRecord } from '../snap';

const affine: [number, number, number][] = [
  [1e-5, 0, -77.43],
  [0, -1e-5, 37.55],
];

// Shaped like richmond p311 in the 2026-08-28 baseline: a fitted page whose
// alias outscores the true-location candidate and the truth pose (#325).
const record: SnapRecord = {
  target: 'p311',
  status: 'ok',
  fit_state: 'fitted',
  width: 1412,
  height: 2037,
  has_truth: true,
  margin: 0.0464,
  incumbent: {
    world_affine: affine,
    verification: -0.708,
    name: { score: 0.1728, n_labels: 9, n_hits: 4 },
    effective_gcps: 2,
    rmse_ft: 210.3,
  },
  truth_pose: {
    world_affine: affine,
    verification: 0.1163,
    inlier_frac: 0.3365,
    ncc_fine: 0.3551,
    chamfer_mean_m: 17.26,
    name: { score: 0.4279, n_labels: 9, n_hits: 5 },
    select_score: 0.6776,
  },
  candidates: [
    {
      world_affine: affine,
      center: [-77.47, 37.57],
      select_score: 1.0207,
      verification: 0.8249,
      rmse_ft: 14713.1,
      theta_deg: -43.96,
      scale_source: 'volume-median',
    },
    {
      world_affine: affine,
      center: [-77.43, 37.55],
      select_score: 0.9743,
      verification: 0.2288,
      rmse_ft: 18.8,
      theta_deg: 52.34,
      scale_source: 'volume-median',
    },
  ],
  decision: {
    path: 'challenge',
    page_verdict: 'keep',
    bars: [
      {
        rule: 'challenge/select',
        need: '>= 1.6',
        got: 1.0207,
        verdict: 'fail',
      },
      {
        rule: 'challenge/name-parity',
        need: "candidate name score >= incumbent's",
        got: 0,
        verdict: 'fail',
        note: 'incumbent 0.1728',
      },
      {
        rule: 'rung-flip',
        need: 'double-scale candidate over a half-scale incumbent',
        got: 'no flip',
        verdict: 'n/a',
      },
    ],
    skipped: [
      {
        rule: 'volume-energy',
        reason: 'volume-level committee may still accept an abstained page',
      },
    ],
  },
};

function render(rec: SnapRecord, selected: number | null = null): string {
  return renderToStaticMarkup(
    <SnapPanel
      record={rec}
      selected={selected}
      onSelect={() => {}}
      onClose={() => {}}
    />,
  );
}

describe('SnapPanel phase-2 records', () => {
  it('renders the truth pose as a row and classifies the outcome', () => {
    const html = render(record);
    expect(html).toContain('<td>truth</td>');
    expect(html).toContain('0.68'); // the truth row's select score
    expect(html).toContain(
      'a pose 14713 ft from truth outscores the truth pose',
    );
  });

  it('renders the decision trace with need/got/verdict lines', () => {
    const html = render(record);
    expect(html).toContain('decision · challenge → keep');
    expect(html).toContain('✗ challenge/select: need &gt;= 1.6, got 1.021');
    expect(html).toContain('got 0 (incumbent 0.1728)');
    expect(html).toContain('– rung-flip: need');
    expect(html).toContain('volume-energy: volume-level committee');
  });

  it('highlights the truth row when it is selected (index -2)', () => {
    expect(render(record, -2)).toMatch(
      /<tr class="selected[^"]*"[^>]*><td>truth<\/td>/,
    );
  });

  it('says when truth exists but was never scored', () => {
    const { truth_pose: _omit, ...unscored } = record;
    const html = render(unscored);
    expect(html).not.toContain('<td>truth</td>');
    expect(html).toContain('truth pose not scored in this record');
  });
});
