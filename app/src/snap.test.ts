import { describe, expect, it } from 'vitest';
import { parseSnapRecords, rankedCandidates } from './snap';

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
