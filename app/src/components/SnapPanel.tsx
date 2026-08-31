import { rankedCandidates } from '../snap';
import type { SnapCandidate, SnapRecord } from '../snap';

/**
 * Per-page snap explorer (#325 phase 1): what the matcher searched, every
 * pose it weighed (winners AND losers), and each pose's evidence components.
 * Selecting a row overlays that pose's P(road) map on the volume map.
 *
 * Everything shown is read from candidates.jsonl — recorded by the pipeline's
 * own scoring; nothing is recomputed here.
 */

interface SnapPanelProps {
  record: SnapRecord;
  /** Index into rankedCandidates(record), -1 for the incumbent, null for none. */
  selected: number | null;
  onSelect: (index: number | null) => void;
  onClose: () => void;
}

// A component bar: label, value in [0, 1]-ish range, red for penalties.
function EvidenceBar({
  label,
  value,
  max,
  penalty,
  detail,
}: {
  label: string;
  value: number | undefined;
  max: number;
  penalty?: boolean;
  detail?: string;
}) {
  if (value == null) return null;
  const frac = Math.max(0, Math.min(1, Math.abs(value) / max));
  return (
    <div className="snap-bar-row" title={detail}>
      <span className="snap-bar-label">{label}</span>
      <span className="snap-bar-track">
        <span
          className={
            penalty ? 'snap-bar-fill snap-bar-penalty' : 'snap-bar-fill'
          }
          style={{ width: `${Math.round(frac * 100)}%` }}
        />
      </span>
      <span className="snap-bar-value">{value.toFixed(3)}</span>
    </div>
  );
}

function CandidateRow({
  label,
  candidate,
  isSelected,
  isBest,
  onClick,
}: {
  label: string;
  candidate: SnapCandidate | (SnapRecord['incumbent'] & object);
  isSelected: boolean;
  isBest: boolean;
  onClick: () => void;
}) {
  const c = candidate as SnapCandidate;
  return (
    <tr
      className={(isSelected ? 'selected ' : '') + (isBest ? 'snap-best' : '')}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      <td>{label}</td>
      <td className="num">{c.select_score?.toFixed(2) ?? '—'}</td>
      <td className="num">{c.verification?.toFixed(2) ?? '—'}</td>
      <td className="num">{c.inlier_frac?.toFixed(2) ?? '—'}</td>
      <td className="num">{c.ncc_fine?.toFixed(2) ?? '—'}</td>
      <td className="num">
        {c.chamfer_mean_m != null ? `${c.chamfer_mean_m.toFixed(1)}m` : '—'}
      </td>
      <td className="num">
        {c.theta_deg != null ? `${c.theta_deg.toFixed(1)}°` : '—'}
      </td>
      <td>{c.scale_source ?? '—'}</td>
      <td className="num">
        {c.rmse_ft != null ? `${c.rmse_ft.toFixed(0)}ft` : ''}
      </td>
    </tr>
  );
}

export function SnapPanel({
  record,
  selected,
  onSelect,
  onClose,
}: SnapPanelProps) {
  const ranked = rankedCandidates(record);
  const active =
    selected === null
      ? null
      : selected === -1
        ? (record.incumbent ?? null)
        : (ranked[selected] ?? null);
  const activeCandidate = active as SnapCandidate | null;

  return (
    <div className="snap-panel">
      <div className="snap-panel-header">
        <strong>snap · {record.target}</strong>
        <span className="snap-meta">
          fit_state {record.fit_state} · status {record.status}
          {record.elapsed_s !== undefined && ` · ${record.elapsed_s}s`}
        </span>
        <button onClick={onClose}>close</button>
      </div>

      {record.search && (
        <p className="snap-search">
          search: {record.search.centers.length} center
          {record.search.centers.length === 1 ? '' : 's'} · radius{' '}
          {Math.round(record.search.radius_m)} m ({record.search.radius_source})
          {record.search.demoted_seed && ' · demoted-pose seed'}
          {' · priors: '}
          {(record.priors?.rotation ?? [])
            .map(
              (p) => `${p.theta_deg.toFixed(0)}°±${p.sigma_deg} (${p.source})`,
            )
            .join(', ') || 'none'}
        </p>
      )}

      <table className="snap-table">
        <thead>
          <tr>
            <th>pose</th>
            <th>select</th>
            <th>verif</th>
            <th>inlier</th>
            <th>ncc</th>
            <th>chamfer</th>
            <th>θ</th>
            <th>scale src</th>
            <th>truth</th>
          </tr>
        </thead>
        <tbody>
          {record.incumbent && (
            <CandidateRow
              label="incumbent"
              candidate={record.incumbent}
              isSelected={selected === -1}
              isBest={false}
              onClick={() => onSelect(selected === -1 ? null : -1)}
            />
          )}
          {ranked.map((candidate, i) => (
            <CandidateRow
              key={i}
              label={`C${i}${candidate.plausible === false ? ' ✗' : ''}`}
              candidate={candidate}
              isSelected={selected === i}
              isBest={i === 0}
              onClick={() => onSelect(selected === i ? null : i)}
            />
          ))}
        </tbody>
      </table>
      {record.has_truth && (
        <p className="snap-note">
          truth column is each pose's grid RMSE; a truth-pose row (scored with
          the same evidence) lands with the phase-2 pipeline record.
        </p>
      )}

      {activeCandidate && (
        <div className="snap-evidence">
          <EvidenceBar
            label="inlier_frac"
            value={activeCandidate.inlier_frac}
            max={1}
            detail="share of P(road) pixels within the chamfer inlier distance of OSM"
          />
          <EvidenceBar
            label="ncc_fine"
            value={activeCandidate.ncc_fine}
            max={1}
            detail="fine-scale normalized cross-correlation, P(road) vs OSM raster"
          />
          <EvidenceBar
            label="chamfer"
            value={activeCandidate.chamfer_mean_m}
            max={30}
            penalty
            detail="mean P(road)→OSM distance in metres (penalty)"
          />
          <EvidenceBar
            label="verification"
            value={activeCandidate.verification}
            max={1.5}
            detail="inlier_frac + ncc_fine − chamfer penalty"
          />
          <EvidenceBar
            label="select"
            value={activeCandidate.select_score}
            max={3}
            detail="the ranking score gates compare against (rescue bar 1.25)"
          />
          {(activeCandidate.gate_reasons?.length ?? 0) > 0 && (
            <p className="snap-note">
              gates: {activeCandidate.gate_reasons!.join(' · ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
