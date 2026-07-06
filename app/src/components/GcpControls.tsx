import type { IntersectionPoint } from '../types';

/** Precomputed stats for the selected seed pair, derived from the georef's `gcp_pairs`. */
export interface GcpFitStats {
  /** True when the pair is coincident/singular (no fit; map stays on the pipeline fit). */
  degenerate: boolean;
  numInliers: number;
  numOutliers: number;
  /** RANSAC score: numInliers * pos_threshold − total inlier positional error (degrees). */
  score: number;
  meanErrorM: number | null;
  maxErrorM: number | null;
}

interface GcpControlsProps {
  intersections: IntersectionPoint[];
  /** Currently chosen seed pair (indices into `intersections`). */
  selectedPair: [number, number];
  onChange: (pair: [number, number]) => void;
  /** The pair chosen by the Python pipeline (initial: true), for the reset button. */
  defaultPair: [number, number] | null;
  /** Precomputed stats for the current pair, or null when the pair has no record. */
  result: GcpFitStats | null;
}

// One-line description of an intersection GCP for the dropdowns.
function gcpLabel(ix: IntersectionPoint, index: number): string {
  return `${index}: ${ix.label_a} × ${ix.label_b}`;
}

/**
 * Georef-mode controls for exploring the seed GCP pair: two dropdowns to pick the
 * intersections that define the fit, a reset to the pipeline's choice, and a readout of the
 * chosen pair's inlier/outlier counts, error, and RANSAC score. Every value shown is
 * precomputed by the Python fitter (`--debug`); this component never re-fits or re-scores.
 */
export function GcpControls(props: GcpControlsProps) {
  const { intersections, selectedPair, onChange, defaultPair, result } = props;
  const [a, b] = selectedPair;

  const isDefault =
    defaultPair !== null &&
    ((defaultPair[0] === a && defaultPair[1] === b) ||
      (defaultPair[0] === b && defaultPair[1] === a));

  return (
    <div className="gcp-controls">
      <div className="gcp-controls-title">Seed GCP pair</div>
      {defaultPair === null && (
        <div className="gcp-nofit-note">
          Page not georeferenced — the pipeline accepted no pair. Showing the
          best-scoring candidate; explore the others below.
        </div>
      )}
      <div className="gcp-select-row">
        <label htmlFor="gcp-a">A</label>
        <select
          id="gcp-a"
          value={a}
          onChange={(e) => onChange([Number(e.target.value), b])}
        >
          {intersections.map((ix, i) => (
            <option key={i} value={i}>
              {gcpLabel(ix, i)}
            </option>
          ))}
        </select>
      </div>
      <div className="gcp-select-row">
        <label htmlFor="gcp-b">B</label>
        <select
          id="gcp-b"
          value={b}
          onChange={(e) => onChange([a, Number(e.target.value)])}
        >
          {intersections.map((ix, i) => (
            <option key={i} value={i}>
              {gcpLabel(ix, i)}
            </option>
          ))}
        </select>
      </div>
      <button
        type="button"
        disabled={defaultPair === null || isDefault}
        onClick={() => defaultPair && onChange(defaultPair)}
      >
        Reset to pipeline pair
      </button>

      {result === null ? (
        <div className="gcp-stats gcp-stats-warning">
          This pair wasn&rsquo;t scored by the pipeline — no fit recorded.
        </div>
      ) : (
        <div className="gcp-stats">
          {result.degenerate ? (
            <div className="gcp-stats-warning">
              Degenerate pair (coincident GCPs) — map unchanged.
            </div>
          ) : (
            <>
              <div>
                <strong>{result.numInliers}</strong> inliers /{' '}
                <strong>{result.numOutliers}</strong> outliers
              </div>
              <div>
                Score: <strong>{result.score.toExponential(3)}</strong>
              </div>
              <div>
                Error (mean / max):{' '}
                {result.meanErrorM !== null
                  ? `${result.meanErrorM.toFixed(1)}m / ${result.maxErrorM?.toFixed(1)}m`
                  : '—'}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
