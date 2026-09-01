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
  name?: number;
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
  name?: number;
  effective_gcps?: number;
  rmse_ft?: number;
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
 * sigma, keeping first-appearance order (rung priority), and aggregate the
 * sources with ×N counts.
 */
export function groupRotationPriors(
  priors: { theta_deg: number; sigma_deg: number; source: string }[],
): string[] {
  const groups = new Map<
    string,
    { label: string; counts: Map<string, number> }
  >();
  for (const prior of priors) {
    let theta = Math.round(prior.theta_deg);
    if (theta === 0) theta = 0; // normalize −0
    if (theta === -180) theta = 180;
    const label = `${theta}°±${prior.sigma_deg}`;
    let group = groups.get(label);
    if (!group) {
      group = { label, counts: new Map() };
      groups.set(label, group);
    }
    group.counts.set(prior.source, (group.counts.get(prior.source) ?? 0) + 1);
  }
  return [...groups.values()].map((group) => {
    const sources = [...group.counts.entries()]
      .map(([source, n]) => (n > 1 ? `${source} ×${n}` : source))
      .join(', ');
    return `${group.label} (${sources})`;
  });
}

/** Candidates sorted by select_score descending (unscored last, order kept). */
export function rankedCandidates(record: SnapRecord): SnapCandidate[] {
  return [...(record.candidates ?? [])].sort(
    (a, b) => (b.select_score ?? -Infinity) - (a.select_score ?? -Infinity),
  );
}
