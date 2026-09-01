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
  score: number;
  n_labels: number;
  n_hits: number;
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
