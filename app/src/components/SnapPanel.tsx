import { groupRotationPriors, rankedCandidates } from '../snap';
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
  const priorGroups = groupRotationPriors(record.priors?.rotation ?? []);

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
          {' · priors:'}
          {priorGroups.length === 0 && ' none'}
        </p>
      )}
      {record.search && priorGroups.length > 0 && (
        <ul className="snap-priors">
          {priorGroups.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      )}

      <table className="snap-table">
        <thead>
          <tr>
            <th>pose</th>
            <th title="the ranking score gates compare against (rescue bar 1.25)">
              select
            </th>
            <th title="inlier_frac + ncc_fine − chamfer penalty">verif</th>
            <th title="share of P(road) pixels within the chamfer inlier distance of OSM">
              inlier
            </th>
            <th title="fine-scale normalized cross-correlation, P(road) vs OSM raster">
              ncc
            </th>
            <th title="mean P(road)→OSM distance in metres (penalty)">
              chamfer
            </th>
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

      {activeCandidate && (activeCandidate.gate_reasons?.length ?? 0) > 0 && (
        <p className="snap-note">
          gates: {activeCandidate.gate_reasons!.join(' · ')}
        </p>
      )}
    </div>
  );
}
