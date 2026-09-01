import {
  groupRotationPriors,
  posePxPerFoot,
  poseRotationDeg,
  rankedCandidates,
  truthVerdict,
} from '../snap';
import type {
  SnapCandidate,
  SnapDecisionBar,
  SnapRecord,
  SnapTruthPose,
} from '../snap';

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
  /** Index into rankedCandidates(record); -1 the incumbent, -2 the truth pose; null none. */
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
  candidate: SnapCandidate | (SnapRecord['incumbent'] & object) | SnapTruthPose;
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
      <td
        className="num"
        title={
          c.theta_deg != null
            ? `ladder seed ${c.theta_deg.toFixed(1)}° (${c.theta_source ?? '?'})`
            : undefined
        }
      >
        {c.world_affine
          ? `${poseRotationDeg(c.world_affine).toFixed(1)}°`
          : c.theta_deg != null
            ? `${c.theta_deg.toFixed(1)}°`
            : '—'}
      </td>
      <td className="num">
        {c.world_affine ? posePxPerFoot(c.world_affine).toFixed(2) : '—'}
      </td>
      <td>{c.scale_source ?? '—'}</td>
      <td className="num">
        {c.rmse_ft != null ? `${c.rmse_ft.toFixed(0)}ft` : ''}
      </td>
    </tr>
  );
}

// One decision bar as a line: "rule: need X, got Y — verdict (note)".
function describeBar(bar: SnapDecisionBar): string {
  const got =
    bar.got === null
      ? '—'
      : typeof bar.got === 'number'
        ? bar.got.toFixed(bar.got === Math.round(bar.got) ? 0 : 3)
        : bar.got;
  const symbol =
    bar.verdict === 'pass' ? '✓' : bar.verdict === 'fail' ? '✗' : '–';
  return `${symbol} ${bar.rule}: need ${bar.need}, got ${got}${bar.note ? ` (${bar.note})` : ''}`;
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
        : selected === -2
          ? (record.truth_pose ?? null)
          : (ranked[selected] ?? null);
  const activeCandidate = active as SnapCandidate | null;
  const priorGroups = groupRotationPriors(record.priors?.rotation ?? []);
  const verdict = truthVerdict(record);

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
            <th title="the pose's rotation in snap's convention (the priors' angle); hover a cell for the ladder seed it started from">
              θ
            </th>
            <th title="the pose's scale in working-frame pixels per foot (the page list's convention)">
              px/ft
            </th>
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
          {record.truth_pose && (
            <CandidateRow
              label="truth"
              candidate={record.truth_pose}
              isSelected={selected === -2}
              isBest={false}
              onClick={() => onSelect(selected === -2 ? null : -2)}
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
      {verdict && (
        <p className={`snap-note snap-truth-${verdict.kind}`}>
          truth: {verdict.detail}
        </p>
      )}
      {record.has_truth && !record.truth_pose && (
        <p className="snap-note">
          truth pose not scored in this record (re-run snap to add it).
        </p>
      )}
      {record.decision && (
        <details className="snap-decision">
          <summary>
            decision · {record.decision.path} → {record.decision.page_verdict}
            {record.decision.argmax_reason &&
              ` (${record.decision.argmax_reason})`}
          </summary>
          <ul>
            {record.decision.bars.map((bar) => (
              <li key={bar.rule} className={`snap-bar-${bar.verdict}`}>
                {describeBar(bar)}
              </li>
            ))}
            {record.decision.skipped.map((skip) => (
              <li key={skip.rule} className="snap-skipped">
                {skip.rule}: {skip.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
      {activeCandidate && (activeCandidate.gate_reasons?.length ?? 0) > 0 && (
        <p className="snap-note">
          gates: {activeCandidate.gate_reasons!.join(' · ')}
        </p>
      )}
    </div>
  );
}
