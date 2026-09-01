import { describe, expect, it } from 'vitest';
import {
  groupRotationPriors,
  parseSnapRecords,
  posePxPerFoot,
  poseRotationDeg,
  rankedCandidates,
  truthVerdict,
} from './snap';

const affine: [number, number, number][] = [
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
  it('groups the fargo p17 ladder by rounded (θ, σ), sorted by angle, with source counts', () => {
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
      '-17°±4 (label-pair-exact ×2, label-osm-mod180 ×2)',
      '-16°±15 (volume-median-theta)',
      '0°±4 (label-pair-exact ×2, label-osm-mod180 ×2)',
      '163°±4 (label-osm-mod180 ×2)',
      '180°±4 (label-osm-mod180 ×2)',
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

describe('truthVerdict', () => {
  const base = {
    target: 'p1',
    status: 'ok',
    fit_state: 'none',
    width: 100,
    height: 80,
  };
  const candidate = (select_score: number, rmse_ft: number) => ({
    world_affine: affine,
    center: [0, 0] as [number, number],
    select_score,
    rmse_ft,
  });

  it('is null without a scored truth pose', () => {
    expect(
      truthVerdict({ ...base, candidates: [candidate(1, 10)] }),
    ).toBeNull();
  });

  it('flags a search problem when truth outscores every candidate', () => {
    const verdict = truthVerdict({
      ...base,
      truth_pose: { world_affine: affine, select_score: 1.8 },
      candidates: [candidate(1.2, 900)],
    });
    expect(verdict?.kind).toBe('search');
    expect(verdict?.detail).toContain('1.80 vs 1.20');
  });

  it('flags an alias when a far-off candidate outscores truth', () => {
    const verdict = truthVerdict({
      ...base,
      truth_pose: { world_affine: affine, select_score: 0.9 },
      candidates: [candidate(1.02, 14713)],
    });
    expect(verdict?.kind).toBe('alias');
    expect(verdict?.detail).toContain('14713 ft');
  });

  it('reports agreement when the top candidate is near truth', () => {
    const verdict = truthVerdict({
      ...base,
      truth_pose: { world_affine: affine, select_score: 0.9 },
      candidates: [candidate(1.1, 12)],
    });
    expect(verdict?.kind).toBe('agrees');
  });

  it('keeps truth_pose and decision through parsing', () => {
    const line = JSON.stringify({
      ...base,
      truth_pose: {
        world_affine: affine,
        select_score: 1.5,
        verification: null,
      },
      decision: {
        path: 'rescue',
        page_verdict: 'abstain',
        bars: [{ rule: 'select', need: '>= 1.35', got: 1.0, verdict: 'fail' }],
        skipped: [],
      },
    });
    const record = parseSnapRecords(line).get('p1')!;
    expect(record.truth_pose?.select_score).toBe(1.5);
    expect('verification' in record.truth_pose!).toBe(false);
    expect(record.decision?.bars[0]?.verdict).toBe('fail');
  });
});

describe('pose rotation and scale', () => {
  // ~0.6 m/px at 40°N, north-up: a00 = 0.6/kx, a11 = -0.6/ky.
  const kx = 111_320 * Math.cos((40 * Math.PI) / 180);
  const northUp: [number, number, number][] = [
    [0.6 / kx, 0, -74],
    [0, -0.6 / 110_540, 40],
  ];

  it('reads 0° and the metre scale off a north-up affine', () => {
    expect(poseRotationDeg(northUp)).toBeCloseTo(0, 6);
    expect(posePxPerFoot(northUp)).toBeCloseTo(1 / (0.6 * 3.28084), 4);
  });

  it('reads a rotated pose in the ladder convention', () => {
    // snap's theta is atan2(-a10, a00) on the RAW affine entries (the same
    // formula the pipeline uses for truth_pose.theta_deg and the demoted-pose
    // prior), so a 30° pose is one whose raw entries carry that rotation.
    const theta = (30 * Math.PI) / 180;
    const c = 1e-5;
    const rotated: [number, number, number][] = [
      [c * Math.cos(theta), c * Math.sin(theta), -74],
      [-c * Math.sin(theta), -c * Math.cos(theta), 40],
    ];
    expect(poseRotationDeg(rotated)).toBeCloseTo(30, 6);
  });
});
