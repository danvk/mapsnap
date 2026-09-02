/**
 * Parser for `mapsnap compare` sidecar tables (`<annotation>.txt`).
 *
 * Rather than recomputing per-page truth error in the browser, the volume viewer reads the
 * comparison the pipeline already produced. Each generated annotation `<name>.iiif.json` has a
 * `<name>.txt` next to it holding the fixed-width table printed by `mapsnap compare`; this
 * turns its paired rows into structured stats keyed by the generated page's file stem.
 *
 * The table (see compare_iiif_georef.print_table) has a header, a `---` rule, one row per truth
 * page — paired rows carry error metrics, `(no fit)` rows are truth-only — a closing `---`, then
 * summary lines. Two layouts exist, told apart by the header (#267):
 *
 * - current: `page_truth` and `page_gen` columns lead every row — the truth page key and the
 *   generated page key it was paired with (`—` on `(no fit)` rows). They differ exactly when
 *   OIM and our splitter numbered a sheet's panels differently.
 * - legacy (archived runs): a single `Page` column holding the truth key; when the numbering
 *   differed the key was marked `(t)` and the generated key followed in trailing parens.
 *
 * Old runs are never regenerated, so both layouts stay readable.
 */

/** One paired page's truth-comparison error, keyed by the generated page's file stem. */
export interface ComparePageStats {
  /** Generated page key (lowercased file stem), e.g. "p1499l" or "p1499n__2". */
  genPageKey: string;
  rmseFt: number;
  maxFt: number;
  translationFt: number;
  rotationErrorDegrees: number;
  scaleErrorPercent: number;
  /** Shear angle in degrees, when the table reports it (skew° column); else undefined. */
  skewDegrees?: number;
  /** Anisotropy (x/y scale ratio, 1 = isotropic), when reported (aniso column); else undefined. */
  anisotropy?: number;
  /** Truth footprint area in km², when the table carries land columns; else undefined. */
  areaKm2?: number;
  /** Usable-land area in km² (street-proximity weighted) — the score's per-page weight. */
  landKm2?: number;
}

/** Response of GET /iiif-api/compare — paired-page stats plus the table's summary footer. */
export interface CompareResponse {
  pages: ComparePageStats[];
  /**
   * Truth page keys the run never placed — the table's `(no fit)` rows, at the
   * granularity the comparison itself counts, so the viewer's missing rows and the
   * "N/M pages georeferenced" line always agree.
   */
  missing: string[];
  /** The summary block below the table ("N/M = …% pages georeferenced", RMSE stats, …); "" if none. */
  footer: string;
  /**
   * Usable-land km² per truth page key, covering BOTH placed and `(no fit)` rows,
   * when the table carries land columns. With `pages[].rmseFt` this is enough to
   * compute the land-weighted score client-side:
   * (Σ land where rmse ≤ 25 − Σ land where rmse ≥ 200) / Σ land over every key here.
   */
  landKm2ByPage?: Record<string, number>;
}

// Whether a line is a header/rule row of the compare table (not a data row).
function isSeparator(line: string): boolean {
  return /^-{3,}$/.test(line.trim());
}

/** Whether a table header announces the two-key layout (`page_truth page_gen …`). */
export function hasTwoPageColumns(header: string): boolean {
  return /^\s*page_truth\s+page_gen\b/.test(header);
}

/** The truth page key of a `(no fit)` row, or null for any other line. */
export function parseMissingRow(line: string): string | null {
  const trailing = line.match(/\s+\(no fit\)\s*$/);
  if (!trailing) return null;
  const tokens = line.slice(0, trailing.index).trim().split(/\s+/);
  const key = tokens[0];
  return key && !isSeparator(key) ? key : null;
}

/**
 * Usable-land km² per page key over every table row (placed and `(no fit)`),
 * or null when the table predates the land columns. The header is the
 * authority: without `land_km2` there, a row's trailing numeric pair is
 * skew/aniso and must not be misread as land.
 */
export function parseLandByPage(text: string): Record<string, number> | null {
  const lines = text.split('\n');
  if (!lines.some((line) => /area_km2\s+land_km2/.test(line))) return null;
  const land: Record<string, number> = {};
  for (const line of lines) {
    const entry = parseRowLandKm2(line);
    if (entry) land[entry[0]] = entry[1];
  }
  return Object.keys(land).length > 0 ? land : null;
}

// The land pair of one row; only meaningful when the header carries land columns.
// Layout-independent: the truth key leads every row in both layouts, and the land
// pair trails it (a `—` in a two-key row's page_gen column is just another token).
function parseRowLandKm2(line: string): [string, number] | null {
  let body = line;
  const trailing = body.match(/\s+\(([^)]*)\)\s*$/);
  if (trailing) body = body.slice(0, trailing.index);
  const tokens = body.trim().split(/\s+/);
  const key = tokens[0];
  if (!key || isSeparator(key) || tokens.length < 3) return null;
  const land = Number(tokens[tokens.length - 1]);
  const area = Number(tokens[tokens.length - 2]);
  // Land columns always travel as a pair; a lone numeric tail is another column.
  if (!Number.isFinite(land) || !Number.isFinite(area)) return null;
  return [key.toLowerCase(), land];
}

// Parse one data row; returns null for `(no fit)` (truth-only) rows and unparseable lines.
// `twoPageColumns` says which layout the table's header announced (see the module doc).
function parseRow(
  line: string,
  twoPageColumns: boolean,
): ComparePageStats | null {
  let body = line;
  let genKeyOverride: string | null = null;
  // A trailing "(…)" is either "(no fit)" or, in the legacy layout when split numbers
  // disagree, the generated key.
  const trailing = body.match(/\s+\(([^)]*)\)\s*$/);
  if (trailing) {
    if (trailing[1] === 'no fit') return null;
    genKeyOverride = trailing[1] ?? null;
    body = body.slice(0, trailing.index);
  }
  const tokens = body.trim().split(/\s+/);
  if (tokens.length < 2) return null;
  let genPageKey: string;
  let numeric: string[];
  if (twoPageColumns) {
    // page_truth page_gen n_t n_g …
    genPageKey = tokens[1] ?? '';
    numeric = tokens.slice(2);
  } else {
    const disagree = tokens[1] === '(t)';
    numeric = tokens.slice(disagree ? 2 : 1);
    genPageKey = (disagree ? (genKeyOverride ?? tokens[0]) : tokens[0]) ?? '';
  }
  // n_t n_g str int t.px g.px rmse max trans rot scale skew aniso
  const rmseFt = Number(numeric[6]);
  const maxFt = Number(numeric[7]);
  const translationFt = Number(numeric[8]);
  const rotationErrorDegrees = Number(numeric[9]);
  const scaleErrorPercent = Number(numeric[10]);
  if (
    !Number.isFinite(rmseFt) ||
    !Number.isFinite(maxFt) ||
    !Number.isFinite(translationFt) ||
    !Number.isFinite(rotationErrorDegrees) ||
    !Number.isFinite(scaleErrorPercent)
  ) {
    return null;
  }
  // Skew/aniso and the land columns trail the table; absent in older tables,
  // so they never gate the row.
  const skewDegrees = Number(numeric[11]);
  const anisotropy = Number(numeric[12]);
  const areaKm2 = Number(numeric[13]);
  const landKm2 = Number(numeric[14]);
  return {
    genPageKey: genPageKey.toLowerCase(),
    rmseFt,
    maxFt,
    translationFt,
    rotationErrorDegrees,
    scaleErrorPercent,
    ...(Number.isFinite(skewDegrees) ? { skewDegrees } : {}),
    ...(Number.isFinite(anisotropy) ? { anisotropy } : {}),
    ...(Number.isFinite(areaKm2) ? { areaKm2 } : {}),
    ...(Number.isFinite(landKm2) ? { landKm2 } : {}),
  };
}

/**
 * Parse a `mapsnap compare` table, returning the paired pages' error stats.
 *
 * Returns [] when the text is not a compare table (e.g. an unrelated `.txt`).
 * `(no fit)` truth-only rows are reported separately by {@link parseMissingTruthKeys}.
 */
export function parseCompareTxt(text: string): ComparePageStats[] {
  const lines = text.split('\n');
  const header = lines.find((line) => line.trim() !== '');
  if (!header || !header.includes('rmse_ft')) return [];
  const start = lines.findIndex(isSeparator);
  if (start < 0) return [];
  const twoPageColumns = hasTwoPageColumns(header);
  const pages: ComparePageStats[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]!;
    if (isSeparator(line)) break; // end of the data section
    if (line.trim() === '') continue;
    const row = parseRow(line, twoPageColumns);
    if (row) pages.push(row);
  }
  return pages;
}

/**
 * Truth page keys the run never placed: the table's `(no fit)` rows, in table order.
 *
 * The comparison already decides this — including which truth split a generated page
 * matched — so the viewer reads it rather than re-deriving coverage from the two
 * annotations, which is how its missing rows drifted from the "N/M pages
 * georeferenced" line the same file reports.
 */
export function parseMissingTruthKeys(text: string): string[] {
  const lines = text.split('\n');
  const header = lines.find((line) => line.trim() !== '');
  if (!header || !header.includes('rmse_ft')) return [];
  const start = lines.findIndex(isSeparator);
  if (start < 0) return [];
  const keys: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]!;
    if (isSeparator(line)) break;
    const key = parseMissingRow(line);
    if (key) keys.push(key);
  }
  return keys;
}

/**
 * The summary block a `mapsnap compare` table prints below its data rows — the
 * "N/M = …% pages georeferenced" line and the RMSE/translation/rotation stats.
 *
 * It is the text after the table's closing `---` rule (the second separator),
 * with surrounding blank lines trimmed. Returns "" when the text has no such
 * footer (e.g. an unrelated `.txt` or a table without a closing rule).
 */
export function parseCompareFooter(text: string): string {
  const lines = text.split('\n');
  const separators = lines.flatMap((line, i) => (isSeparator(line) ? [i] : []));
  if (separators.length < 2) return '';
  return lines
    .slice(separators[1]! + 1)
    .join('\n')
    .trim();
}
