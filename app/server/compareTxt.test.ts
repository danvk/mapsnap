import { describe, expect, it } from 'vitest';
import { parseCompareFooter, parseCompareTxt } from './compareTxt.ts';

const HEADER =
  'Page          n_t n_g  str  int  t.px/ft  g.px/ft   rmse_ft    max_ft   trans_ft   rot_err  scale_%   skew°   aniso';
const RULE = '-'.repeat(HEADER.length);

// A compare table built from real 2026-07-21.txt rows: a plain pair, a split whose numbering
// disagrees (gen key in trailing parens), and a truth-only "(no fit)" row.
const TABLE = [
  HEADER,
  RULE,
  'p1404__2        5   2    5    6     6.02     5.85      18.2      22.7       15.4     +0.34    +2.88   +0.73   1.030',
  'p1499N__2       5   2    5    5     6.03     6.13      12.8      17.6        7.9     +0.31    -1.68   +0.66   1.024',
  'p1499L__3 (t)  10   2    9    6     2.97     3.00      10.5      16.3        8.6     -0.01    -1.12   -0.26   1.015  (p1499L)',
  'p1401__2        3   —    —    —        —        —         —         —          —         —        —   -2.31   1.009  (no fit)',
  RULE,
  '',
  '111/129 = 86.05% pages georeferenced (18 total losses)',
  'RMSE:  mean=31 ft  median=12 ft  max=403 ft',
].join('\n');

describe('parseCompareTxt', () => {
  it('parses paired rows keyed by the generated page stem', () => {
    const pages = parseCompareTxt(TABLE);
    expect(pages.map((p) => p.genPageKey)).toEqual([
      'p1404__2',
      'p1499n__2', // uppercase suffix lowercased to match the file stem
      'p1499l', // split numbers disagree: generated key comes from the trailing parens
    ]);
  });

  it('reads the error metrics of a plain paired row', () => {
    const p1404 = parseCompareTxt(TABLE).find(
      (p) => p.genPageKey === 'p1404__2',
    )!;
    expect(p1404).toMatchObject({
      rmseFt: 18.2,
      maxFt: 22.7,
      translationFt: 15.4,
      rotationErrorDegrees: 0.34,
      scaleErrorPercent: 2.88,
    });
  });

  it('takes the generated key from the trailing parens when numbering disagrees', () => {
    const p1499l = parseCompareTxt(TABLE).find(
      (p) => p.genPageKey === 'p1499l',
    )!;
    expect(p1499l.rmseFt).toBe(10.5);
  });

  it('drops "(no fit)" truth-only rows', () => {
    expect(
      parseCompareTxt(TABLE).some((p) => p.genPageKey === 'p1401__2'),
    ).toBe(false);
  });

  it('returns [] for a non-compare text file', () => {
    expect(parseCompareTxt('LOS ANGELES\nsome ocr text\n')).toEqual([]);
  });
});

describe('parseCompareFooter', () => {
  it('returns the summary block below the closing rule, trimmed', () => {
    expect(parseCompareFooter(TABLE)).toBe(
      '111/129 = 86.05% pages georeferenced (18 total losses)\n' +
        'RMSE:  mean=31 ft  median=12 ft  max=403 ft',
    );
  });

  it('returns "" when there is no closing rule / summary', () => {
    const noFooter = [HEADER, RULE, TABLE.split('\n')[2]].join('\n');
    expect(parseCompareFooter(noFooter)).toBe('');
    expect(parseCompareFooter('unrelated text file')).toBe('');
  });
});
