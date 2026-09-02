import { describe, expect, it } from 'vitest';
import {
  hasTwoPageColumns,
  parseCompareFooter,
  parseCompareTxt,
  parseLandByPage,
  parseMissingTruthKeys,
} from './compareTxt.ts';

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
      skewDegrees: 0.73,
      anisotropy: 1.03,
    });
  });

  it('omits skew/aniso when the table has no such columns', () => {
    const noSkew =
      'Page  n_t n_g  str  int  t.px/ft  g.px/ft   rmse_ft    max_ft   trans_ft   rot_err  scale_%\n' +
      '-'.repeat(60) +
      '\np1  5   2    5    6     6.02     5.85      18.2      22.7       15.4     +0.34    +2.88';
    const [row] = parseCompareTxt(noSkew);
    expect(row?.rmseFt).toBe(18.2);
    expect(row?.skewDegrees).toBeUndefined();
    expect(row?.anisotropy).toBeUndefined();
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

describe('land columns', () => {
  const header =
    'Page          n_t n_g  str  int  t.px/ft  g.px/ft   rmse_ft    max_ft   trans_ft   rot_err  scale_%   skew\u00b0   aniso   area_km2   land_km2';
  const row =
    'p57             3   2    3    2     4.89     2.64      12.3      20.1        9.4     +0.72   +1.36   -0.56   1.017     0.0622     0.0497';
  const missingRow =
    'p50             4   \u2014   \u2014   \u2014  \u2014  \u2014  \u2014  \u2014  \u2014  \u2014  \u2014  +0.78   1.008     0.0622     0.0497  (no fit)';

  it('parses areaKm2/landKm2 from a paired row', () => {
    const pages = parseCompareTxt([header, '-'.repeat(120), row].join('\n'));
    expect(pages).toHaveLength(1);
    expect(pages[0]!.areaKm2).toBeCloseTo(0.0622);
    expect(pages[0]!.landKm2).toBeCloseTo(0.0497);
  });

  it('collects land for placed and no-fit rows, gated on the header', () => {
    const text = [header, '-'.repeat(120), row, missingRow].join('\n');
    expect(parseLandByPage(text)).toEqual({ p57: 0.0497, p50: 0.0497 });
    // An old-format table (no land columns) must NOT misread skew/aniso.
    const oldHeader = header.replace(/\s+area_km2\s+land_km2/, '');
    const oldRow =
      'p9              3   2    4    3     4.90     4.67      59.2     103.1       47.3     -2.11    +4.89   +0.25   1.019';
    expect(
      parseLandByPage([oldHeader, '-'.repeat(100), oldRow].join('\n')),
    ).toBeNull();
  });
});

// The two-key layout (#267): page_truth and page_gen lead every row. Rows built from a
// real 2026-08-28 richmond table plus the three shapes that used to hide behind "(t)":
// a whole page, a split whose numbering disagrees, a whole truth page we split, and a
// truth-only "(no fit)" row.
const HEADER2 =
  'page_truth    page_gen      n_t n_g  str  int  t.px/ft  g.px/ft   rmse_ft    max_ft   trans_ft   rot_err  scale_%   skew°   aniso   area_km2   land_km2';
const RULE2 = '-'.repeat(HEADER2.length);
const TABLE2 = [
  HEADER2,
  RULE2,
  'p311          p311            9   4    0    0     5.72     5.61      18.8      31.2       10.4    -0.31    +1.84   +2.13   1.067     0.0837     0.0837',
  'p13__1        p13__2         10   2    9    6     2.97     3.00      10.5      16.3        8.6    -0.01    -1.12   -0.26   1.015     0.0412     0.0398',
  'p16N          p16N__1         6   4    0    0     6.12     6.30      60.4     100.9       50.2    +4.88    -2.88   -0.19   1.006     0.0761     0.0761',
  'p12__2        —               3   —    —    —        —        —         —         —          —        —        —   -3.62   1.015     0.0222     0.0176  (no fit)',
  RULE2,
  '',
  '3/4 = 75.00% pages georeferenced (1 total losses)',
].join('\n');

describe('two page columns (#267)', () => {
  it('detects the layout from the header', () => {
    expect(hasTwoPageColumns(HEADER2)).toBe(true);
    expect(hasTwoPageColumns(HEADER)).toBe(false);
  });

  it('keys paired rows by the page_gen column, not the truth key', () => {
    const keys = parseCompareTxt(TABLE2).map((p) => p.genPageKey);
    expect(keys).toEqual(['p311', 'p13__2', 'p16n__1']);
  });

  it('reads the metrics after both key columns', () => {
    const p13 = parseCompareTxt(TABLE2).find((p) => p.genPageKey === 'p13__2')!;
    expect(p13.rmseFt).toBe(10.5);
    expect(p13.maxFt).toBe(16.3);
    expect(p13.translationFt).toBe(8.6);
    expect(p13.rotationErrorDegrees).toBe(-0.01);
    expect(p13.scaleErrorPercent).toBe(-1.12);
    expect(p13.skewDegrees).toBe(-0.26);
    expect(p13.anisotropy).toBe(1.015);
    expect(p13.landKm2).toBe(0.0398);
  });

  it('reports "(no fit)" rows by their truth key and keeps them out of pages', () => {
    expect(parseMissingTruthKeys(TABLE2)).toEqual(['p12__2']);
    expect(parseCompareTxt(TABLE2).some((p) => p.genPageKey === '—')).toBe(
      false,
    );
  });

  it('keys land by the truth page in both layouts', () => {
    const land = parseLandByPage(TABLE2)!;
    expect(land['p13__1']).toBe(0.0398);
    expect(land['p12__2']).toBe(0.0176);
    expect(land['p13__2']).toBeUndefined();
  });

  it('still reads the legacy single-column layout unchanged', () => {
    const legacy = parseCompareTxt(TABLE).find(
      (p) => p.genPageKey === 'p1499l',
    )!;
    expect(legacy.rmseFt).toBe(10.5);
  });
});
