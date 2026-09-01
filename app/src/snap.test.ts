import { describe, expect, it } from 'vitest';
import {
  groupRotationPriors,
  parseSnapRecords,
  rankedCandidates,
} from './snap';

const affine = [
  [1, 0, 0],
  [0, 1, 0],
];

// One candidates.jsonl row with a scored candidate and an unscored one whose
// numeric fields the pipeline wrote as JSON null. Clicking the unscored
// candidate used to crash EvidenceBar (null.toFixed) on fargo p17.
const row = JSON.stringify({
  target: 'p17',
  status: 'ok',
  fit_state: 'fit',
  width: 100,
  height: 80,
  candidates: [
    {
      world_affine: affine,
      center: [0, 0],
      select_score: 1.5,
      verification: 0.8,
    },
    {
      world_affine: affine,
      center: [0, 0],
      select_score: null,
      verification: null,
    },
  ],
});

describe('parseSnapRecords', () => {
  it('strips JSON nulls so optional-number fields are absent, not null', () => {
    const record = parseSnapRecords(row).get('p17')!;
    const unscored = record.candidates![1]!;
    expect('select_score' in unscored).toBe(false);
    expect('verification' in unscored).toBe(false);
    expect(unscored.world_affine).toEqual(affine);
    expect(record.candidates![0]!.select_score).toBe(1.5);
  });

  it('ranks unscored candidates last', () => {
    const record = parseSnapRecords(row).get('p17')!;
    const ranked = rankedCandidates(record);
    expect(ranked[0]!.select_score).toBe(1.5);
    expect(ranked[1]!.select_score).toBeUndefined();
  });
});

describe('groupRotationPriors', () => {
  it('groups the fargo p17 ladder by rounded (θ, σ) with source counts', () => {
    const prior = (theta_deg: number, source: string, sigma_deg = 4) => ({
      theta_deg,
      sigma_deg,
      source,
    });
    const ladder = [
      prior(0.1, 'label-pair-exact'),
      prior(0.2, 'label-pair-exact'),
      prior(-17.4, 'label-pair-exact'),
      prior(-17.2, 'label-pair-exact'),
      prior(0.1, 'label-osm-mod180'),
      prior(-179.9, 'label-osm-mod180'),
      prior(162.6, 'label-osm-mod180'),
      prior(-17.4, 'label-osm-mod180'),
      prior(162.8, 'label-osm-mod180'),
      prior(-17.2, 'label-osm-mod180'),
      prior(180.0, 'label-osm-mod180'),
      prior(-0.2, 'label-osm-mod180'),
      prior(-16.0, 'volume-median-theta', 15),
    ];
    expect(groupRotationPriors(ladder)).toEqual([
      '0°±4 (label-pair-exact ×2, label-osm-mod180 ×2)',
      '-17°±4 (label-pair-exact ×2, label-osm-mod180 ×2)',
      '180°±4 (label-osm-mod180 ×2)',
      '163°±4 (label-osm-mod180 ×2)',
      '-16°±15 (volume-median-theta)',
    ]);
  });

  it('separates identical angles with different sigmas', () => {
    expect(
      groupRotationPriors([
        { theta_deg: 12, sigma_deg: 4, source: 'a' },
        { theta_deg: 12, sigma_deg: 15, source: 'b' },
      ]),
    ).toEqual(['12°±4 (a)', '12°±15 (b)']);
  });
});
