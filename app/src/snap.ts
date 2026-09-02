/**
 * Snap-channel debug data: candidates.jsonl parsing and pose geometry (#325).
 *
 * The snap stage records everything it weighed per page in
 * `data/<vol>/artifacts/osm_snap/candidates.jsonl` — search centers with
 * provenance, rotation/scale priors, every candidate pose with its evidence
 * components, and the incumbent's evaluation. This module reads that record;
 * it deliberately re-implements NO scoring (the compare-output lesson): every
 * number shown in the UI was computed by the pipeline itself.
 */

/** OCR street-name agreement with a pose (a boost, never a gate). */
export interface SnapNameAlignment {
  /** Reward-only alignment: Σ exp(−d/25 m) over hits, / (n_labels + 2). */
  score: number;
  n_labels: number;
  n_hits: number;
  /** The signed term the ranking uses (#375); absent on older records. */
  evidence?: number;
  hits?: { text: string; dist_m: number; angle_deg: number }[];
}

/** One candidate pose from the snap matcher, as recorded in candidates.jsonl. */
export interface SnapCandidate {
  world_affine: [number, number, number][];
  center: [number, number];
  center_dist_m?: number;
  theta_deg?: number;
  theta_source?: string;
  scale_m_per_px?: number;
  scale_source?: string;
  scale_adjust?: number;
  ncc?: number;
  ncc_fine?: number;
  chamfer_mean_m?: number;
  inlier_frac?: number;
  n_points?: number;
  overlap_frac?: number;
  region_containment?: number;
  /** Sigmas from the nearest directed rotation prior; absent without one. */
  prior_theta_residual_sigma?: number;
  refine_shift_m?: number;
  select_score?: number;
  verification?: number;
  name?: SnapNameAlignment;
  plausible?: boolean;
  gate_reasons?: string[];
  /** Grid RMSE vs truth, present only when the volume has truth data. */
  rmse_ft?: number;
}

/** The incumbent (published RANSAC pose) evaluated with the same evidence. */
export interface SnapIncumbent {
  world_affine: [number, number, number][];
  verification?: number;
  ncc_fine?: number;
  inlier_frac?: number;
  chamfer_mean_m?: number;
  n_points?: number;
  name?: SnapNameAlignment;
  effective_gcps?: number;
  rmse_ft?: number;
}

/** The truth pose scored with the ladder's own evidence (#325 phase 2). */
export interface SnapTruthPose extends SnapIncumbent {
  select_score?: number;
  theta_deg?: number;
  region_containment?: number;
  prior_theta_residual_sigma?: number;
}

/** One need/got/verdict line of the pipeline's decision trace. */
export interface SnapDecisionBar {
  rule: string;
  need: string;
  got: number | string | null;
  verdict: 'pass' | 'fail' | 'n/a';
  note?: string;
}

/** The per-page decision trace recorded at snap time (#325 phase 2). */
export interface SnapDecision {
  path: string;
  page_verdict: string;
  bars: SnapDecisionBar[];
  skipped: { rule: string; reason: string }[];
  argmax_reason?: string;
}

/** One page's full snap record. */
export interface SnapRecord {
  target: string;
  status: string;
  fit_state: string;
  width: number;
  height: number;
  elapsed_s?: number;
  margin?: number;
  has_truth?: boolean;
  search?: {
    centers: [number, number][];
    radius_m: number;
    radius_source: string;
    demoted_seed?: boolean;
  };
  priors?: {
    rotation: { theta_deg: number; sigma_deg: number; source: string }[];
    scale: { m_per_px: number; sigma_log: number; source: string }[];
  };
  incumbent?: SnapIncumbent;
  truth_pose?: SnapTruthPose;
  decision?: SnapDecision;
  candidates?: SnapCandidate[];
}

// Recursively drop null-valued keys: the pipeline writes JSON null for
// unscored fields (e.g. an unverifiable candidate's select_score), but
// consumers type these as optional numbers and guard with undefined checks.
function stripNulls(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripNulls);
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
      if (entry !== null) out[key] = stripNulls(entry);
    }
    return out;
  }
  return value;
}

/** Parse candidates.jsonl text into a stem-keyed map (later rows win). */
export function parseSnapRecords(jsonl: string): Map<string, SnapRecord> {
  const records = new Map<string, SnapRecord>();
  for (const line of jsonl.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const record = stripNulls(JSON.parse(trimmed)) as SnapRecord;
      if (record.target) records.set(record.target, record);
    } catch {
      // A truncated final line (interrupted run) is not an error worth surfacing.
    }
  }
  return records;
}

/**
 * The four page corners of a pose in lon/lat, TL/TR/BR/BL — the coordinate
 * order maplibre image sources expect. `world_affine` maps page px → lon/lat.
 */
export function poseCorners(
  affine: [number, number, number][],
  width: number,
  height: number,
): [[number, number], [number, number], [number, number], [number, number]] {
  const apply = (x: number, y: number): [number, number] => [
    affine[0][0] * x + affine[0][1] * y + affine[0][2],
    affine[1][0] * x + affine[1][1] * y + affine[1][2],
  ];
  return [apply(0, 0), apply(width, 0), apply(width, height), apply(0, height)];
}

/**
 * Rotation priors grouped for display: one line per distinct (θ, σ).
 *
 * The pipeline records one prior per vote (per label, per label pair), so the
 * raw ladder is full of repeats — the multiplicity is the corroboration
 * signal. Group by the rounded angle (−0 → 0, −180 → 180: same rotation) and
 * sigma, sorted by angle then sigma, and aggregate the sources with ×N counts
 * (in first-appearance order, which is rung priority).
 */
export function groupRotationPriors(
  priors: { theta_deg: number; sigma_deg: number; source: string }[],
): string[] {
  const groups = new Map<
    string,
    { theta: number; sigma: number; counts: Map<string, number> }
  >();
  for (const prior of priors) {
    let theta = Math.round(prior.theta_deg);
    if (theta === 0) theta = 0; // normalize −0
    if (theta === -180) theta = 180;
    const label = `${theta}°±${prior.sigma_deg}`;
    let group = groups.get(label);
    if (!group) {
      group = { theta, sigma: prior.sigma_deg, counts: new Map() };
      groups.set(label, group);
    }
    group.counts.set(prior.source, (group.counts.get(prior.source) ?? 0) + 1);
  }
  return [...groups.values()]
    .sort((a, b) => a.theta - b.theta || a.sigma - b.sigma)
    .map((group) => {
      const sources = [...group.counts.entries()]
        .map(([source, n]) => (n > 1 ? `${source} ×${n}` : source))
        .join(', ');
      return `${group.theta}°±${group.sigma} (${sources})`;
    });
}

const FEET_PER_METER = 3.28084;

/**
 * A pose's rotation in degrees, in snap's own convention (the angle the
 * rotation ladder and the priors use: atan2(-a10, a00) of the page->lon/lat
 * affine, 0 for a north-up page). Derived from the pose itself, so it is the
 * refined pose's rotation — a candidate's recorded `theta_deg` is only the
 * ladder seed it started from.
 */
export function poseRotationDeg(affine: [number, number, number][]): number {
  const [row0, row1] = affine;
  return (Math.atan2(-(row1?.[0] ?? 0), row0?.[0] ?? 1) * 180) / Math.PI;
}

/**
 * A pose's scale in pixels per foot, in the frame of the pixels the affine
 * maps (snap's working-scale page), which is the page list's convention.
 */
export function posePxPerFoot(affine: [number, number, number][]): number {
  const [row0, row1] = affine;
  const lat = row1?.[2] ?? 0;
  const kx = 111_320 * Math.cos((lat * Math.PI) / 180);
  const ky = 110_540;
  const metresPerPx = Math.hypot((row0?.[0] ?? 0) * kx, (row1?.[0] ?? 0) * ky);
  return metresPerPx > 0 ? 1 / (metresPerPx * FEET_PER_METER) : NaN;
}

/** Candidates sorted by select_score descending (unscored last, order kept). */
export function rankedCandidates(record: SnapRecord): SnapCandidate[] {
  return [...(record.candidates ?? [])].sort(
    (a, b) => (b.select_score ?? -Infinity) - (a.select_score ?? -Infinity),
  );
}

/** How the truth pose fared against the ladder: the one-line failure class. */
export interface TruthVerdict {
  kind: 'search' | 'alias' | 'agrees';
  detail: string;
}

/**
 * Classify a page's snap outcome against its scored truth pose (#325).
 *
 * Truth outscoring every candidate means the search never reached the right
 * pose (a search problem); a candidate far from truth outscoring it means the
 * page's own evidence prefers a wrong pose (a data/aliasing problem); a top
 * candidate near truth means the evidence agrees. Null without a scored truth.
 */
export function truthVerdict(record: SnapRecord): TruthVerdict | null {
  const truth = record.truth_pose;
  if (!truth || truth.select_score === undefined) return null;
  const top = rankedCandidates(record).find(
    (c) => c.select_score !== undefined,
  );
  const truthScore = truth.select_score.toFixed(2);
  if (!top || top.select_score === undefined) {
    return {
      kind: 'search',
      detail: `no plausible candidate; the truth pose scores ${truthScore}`,
    };
  }
  const topScore = top.select_score.toFixed(2);
  if (truth.select_score > top.select_score) {
    return {
      kind: 'search',
      detail: `truth pose outscores every candidate (${truthScore} vs ${topScore}): the search never reached it`,
    };
  }
  const rmse = top.rmse_ft;
  if (rmse !== undefined && rmse > 200) {
    return {
      kind: 'alias',
      detail: `a pose ${Math.round(rmse)} ft from truth outscores the truth pose (${topScore} vs ${truthScore}): the evidence prefers an alias`,
    };
  }
  return {
    kind: 'agrees',
    detail: `top candidate is ${rmse === undefined ? 'near' : `${Math.round(rmse)} ft from`} truth and outscores it (${topScore} vs ${truthScore})`,
  };
}

// The ranking weights and the chamfer clamp, mirrored from the pipeline
// (mapsnap/osm_snap.py W_NAME / W_CONTAIN / W_PRIOR, mapsnap/edge_join.py
// CHAMFER_CLAMP_M). Records carry the terms and the totals but not the
// weights; selectBreakdown flags a total that no longer adds up, which is how
// a weight change upstream shows itself here.
export const W_NAME = 1.0;
export const W_CONTAIN = 0.3;
export const W_PRIOR = 0.1;
export const CHAMFER_CLAMP_M = 30.0;
/** Records round scores to 4 decimals; a larger gap means the weights moved. */
const BREAKDOWN_TOLERANCE = 0.01;

/** One additive term of a score: its label, the product it contributes, how it was formed. */
export interface ScoreTerm {
  label: string;
  /** The signed contribution to the total; null when the record lacks the input. */
  value: number | null;
  /** How the value was formed, or why it is missing. */
  detail: string;
}

/** A score decomposed into the terms recorded for the pose. */
export interface ScoreBreakdown {
  /** The score name: "select" or "verif". */
  label: string;
  terms: ScoreTerm[];
  /** Sum of the present terms; null when the base term is missing. */
  total: number | null;
  /** The score as the pipeline recorded it; null when not recorded. */
  recorded: number | null;
  /** Set when the terms do not add up to the recorded score. */
  note?: string;
}

type Scored = {
  verification?: number;
  select_score?: number;
  inlier_frac?: number;
  ncc_fine?: number;
  chamfer_mean_m?: number;
  name?: SnapNameAlignment;
  region_containment?: number;
  prior_theta_residual_sigma?: number;
};

// Sum the present terms; null when the first (base) term is missing.
function sumTerms(terms: ScoreTerm[]): number | null {
  if (terms[0].value === null) return null;
  return terms.reduce((acc, term) => acc + (term.value ?? 0), 0);
}

// The mismatch note, when the terms and the recorded total disagree.
function mismatchNote(
  total: number | null,
  recorded: number | null,
): string | undefined {
  if (total === null || recorded === null) return undefined;
  if (Math.abs(total - recorded) <= BREAKDOWN_TOLERANCE) return undefined;
  return `terms sum to ${total.toFixed(3)}, recorded ${recorded.toFixed(3)}: the pipeline's weights differ from this view's`;
}

/**
 * The matcher's verification score decomposed: inlier_frac + ncc_fine −
 * chamfer_mean_m / CHAMFER_CLAMP_M (edge_join.JoinCandidate.verification_score).
 */
export function verificationBreakdown(pose: Scored): ScoreBreakdown {
  const terms: ScoreTerm[] = [
    {
      label: 'inlier',
      value: pose.inlier_frac ?? null,
      detail:
        pose.inlier_frac != null
          ? 'share of P(road) pixels within the inlier distance of OSM'
          : 'not recorded',
    },
    {
      label: 'ncc',
      value: pose.ncc_fine ?? null,
      detail:
        pose.ncc_fine != null
          ? 'fine-scale correlation, P(road) vs OSM'
          : 'not recorded',
    },
    {
      label: 'chamfer',
      value:
        pose.chamfer_mean_m != null
          ? -pose.chamfer_mean_m / CHAMFER_CLAMP_M
          : null,
      detail:
        pose.chamfer_mean_m != null
          ? `−${pose.chamfer_mean_m.toFixed(1)} m / ${CHAMFER_CLAMP_M} m`
          : 'not recorded',
    },
  ];
  const total = terms.every((term) => term.value !== null)
    ? sumTerms(terms)
    : null;
  const recorded = pose.verification ?? null;
  return {
    label: 'verif',
    terms,
    total,
    recorded,
    note: mismatchNote(total, recorded),
  };
}

/**
 * The ranking score decomposed: verification plus the three soft-evidence
 * terms (osm_snap.selection_score). A term the record lacks is shown as
 * missing and contributes nothing, exactly as the pipeline skips a None term.
 */
export function selectBreakdown(pose: Scored): ScoreBreakdown {
  const name = pose.name;
  let nameTerm: ScoreTerm;
  if (!name) {
    nameTerm = {
      label: 'name',
      value: null,
      detail: 'no street labels matched to OSM',
    };
  } else {
    const evidence = name.evidence ?? name.score;
    const hits = `${name.n_hits}/${name.n_labels} labels hit`;
    const basis =
      name.evidence != null
        ? `signed evidence ${evidence.toFixed(3)}`
        : `reward-only score ${evidence.toFixed(3)} (record predates the signed term)`;
    nameTerm = {
      label: 'name',
      value: W_NAME * evidence,
      detail: `${W_NAME} × ${basis}; ${hits}`,
    };
  }
  const containment = pose.region_containment;
  const containTerm: ScoreTerm =
    containment != null
      ? {
          label: 'containment',
          value: W_CONTAIN * containment,
          detail: `${W_CONTAIN} × ${(containment * 100).toFixed(0)}% of the footprint inside the key-map region`,
        }
      : {
          label: 'containment',
          value: null,
          detail: 'no key-map region for this page',
        };
  const sigma = pose.prior_theta_residual_sigma;
  const priorTerm: ScoreTerm =
    sigma != null
      ? {
          label: 'prior',
          value: W_PRIOR * Math.max(-1, 1 - sigma),
          detail: `${W_PRIOR} × max(−1, 1 − ${sigma.toFixed(2)}σ from the nearest directed rotation prior)`,
        }
      : { label: 'prior', value: null, detail: 'no directed rotation prior' };
  const terms: ScoreTerm[] = [
    {
      label: 'verif',
      value: pose.verification ?? null,
      detail:
        pose.verification != null
          ? 'matcher verification (see its own breakdown)'
          : 'not recorded (implausible pose scores −∞)',
    },
    nameTerm,
    containTerm,
    priorTerm,
  ];
  const total = sumTerms(terms);
  const recorded = pose.select_score ?? null;
  return {
    label: 'select',
    terms,
    total,
    recorded,
    note: mismatchNote(total, recorded),
  };
}
