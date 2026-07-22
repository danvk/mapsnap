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
 * summary lines. A paired row's Page column is the truth page key; when our split numbering
 * differs it is marked `(t)` and the generated key follows in trailing parens.
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
}

/** Response of GET /iiif-api/compare — paired-page stats from the sidecar table. */
export interface CompareResponse {
  pages: ComparePageStats[];
}

// Whether a line is a header/rule row of the compare table (not a data row).
function isSeparator(line: string): boolean {
  return /^-{3,}$/.test(line.trim());
}

// Parse one data row; returns null for `(no fit)` (truth-only) rows and unparseable lines.
function parseRow(line: string): ComparePageStats | null {
  let body = line;
  let genKeyOverride: string | null = null;
  // A trailing "(…)" is either "(no fit)" or, when split numbers disagree, the generated key.
  const trailing = body.match(/\s+\(([^)]*)\)\s*$/);
  if (trailing) {
    if (trailing[1] === 'no fit') return null;
    genKeyOverride = trailing[1] ?? null;
    body = body.slice(0, trailing.index);
  }
  const tokens = body.trim().split(/\s+/);
  if (tokens.length < 2) return null;
  const disagree = tokens[1] === '(t)';
  const numeric = tokens.slice(disagree ? 2 : 1);
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
  const genPageKey =
    (disagree ? (genKeyOverride ?? tokens[0]) : tokens[0]) ?? '';
  return {
    genPageKey: genPageKey.toLowerCase(),
    rmseFt,
    maxFt,
    translationFt,
    rotationErrorDegrees,
    scaleErrorPercent,
  };
}

/**
 * Parse a `mapsnap compare` table, returning the paired pages' error stats.
 *
 * Returns [] when the text is not a compare table (e.g. an unrelated `.txt`). Only rows with a
 * fit are returned; `(no fit)` truth-only rows are dropped (the viewer sources missing pages
 * from the truth annotation instead).
 */
export function parseCompareTxt(text: string): ComparePageStats[] {
  const lines = text.split('\n');
  const header = lines.find((line) => line.trim() !== '');
  if (!header || !header.includes('rmse_ft')) return [];
  const start = lines.findIndex(isSeparator);
  if (start < 0) return [];
  const pages: ComparePageStats[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]!;
    if (isSeparator(line)) break; // end of the data section
    if (line.trim() === '') continue;
    const row = parseRow(line);
    if (row) pages.push(row);
  }
  return pages;
}
