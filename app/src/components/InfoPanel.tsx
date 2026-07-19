import type { SkippedItem } from '../../server/iiifAnnotations';
import type { PageCompareStats } from '../iiif/compare';
import type { PageGeo } from '../iiif/pages';

interface InfoPanelProps {
  /** All pages in the loaded annotation, or [] before one is loaded. */
  pages: PageGeo[];
  /** Items the server dropped while rewriting the annotation. */
  skipped: SkippedItem[];
  /** The loaded annotation file's name, for the summary header. */
  annotationName: string | null;
  selectedPage: PageGeo | null;
  /** Truth-compare stats for the selected page, when the volume has truth. */
  selectedStats: PageCompareStats | null;
  /** The selected page's note text, or null when it has none. */
  selectedNote: string | null;
  /** Volume directory name, e.g. "brooklyn_ny_1906_vol_6". */
  volume: string;
  onClose: () => void;
}

// Median of a non-empty list of numbers.
function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]!
    : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

// Fit-type counts like "70 polynomial, 7 helmert", most common first.
function fitSummary(pages: PageGeo[]): string {
  const counts = new Map<string, number>();
  for (const page of pages) {
    counts.set(
      page.transformationType,
      (counts.get(page.transformationType) ?? 0) + 1,
    );
  }
  return Array.from(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([type, count]) => `${count} ${type}`)
    .join(', ');
}

/**
 * The volume viewer's side panel: stats and debugger links for the selected
 * page, or a summary of the loaded Georeference Annotation file when nothing
 * is selected. Always rendered, so selecting a page doesn't resize the map.
 *
 * The links use the debugger's `?files=` deep-link convention, so they open
 * the page's streets or georef view in this same app.
 */
export function InfoPanel(props: InfoPanelProps) {
  const {
    pages,
    skipped,
    annotationName,
    selectedPage,
    selectedStats,
    selectedNote,
    volume,
    onClose,
  } = props;

  if (selectedPage) {
    const base = `data/${volume}/${selectedPage.pageKey}`;
    return (
      <div className="page-info-panel">
        <div className="page-info-header">
          <strong>{selectedPage.pageKey}</strong>
          <button type="button" onClick={onClose} title="Deselect page">
            ×
          </button>
        </div>
        <dl>
          {selectedStats && (
            <>
              <dt>RMSE</dt>
              <dd>
                {selectedStats.rmseFt.toFixed(1)} ft (max{' '}
                {selectedStats.maxFt.toFixed(1)} ft)
              </dd>
              <dt>Translation</dt>
              <dd>{selectedStats.translationFt.toFixed(1)} ft</dd>
              <dt>Rotation Δ</dt>
              <dd>{selectedStats.rotationErrorDegrees.toFixed(2)}°</dd>
              <dt>Scale Δ</dt>
              <dd>{selectedStats.scaleErrorPercent.toFixed(2)}%</dd>
            </>
          )}
          <dt>Scale</dt>
          <dd>{selectedPage.scalePixelsPerFoot.toFixed(2)} px/ft</dd>
          <dt>Rotation</dt>
          <dd>{selectedPage.rotationDegrees.toFixed(1)}°</dd>
          <dt>Size</dt>
          <dd>
            {selectedPage.width} × {selectedPage.height} px
          </dd>
          <dt>GCPs</dt>
          <dd>{selectedPage.gcps.length}</dd>
          <dt>Fit</dt>
          <dd>{selectedPage.transformationType}</dd>
        </dl>
        {selectedNote && (
          <div className="page-info-note">
            <span className="page-info-note-label">📓 Note</span>
            <p>{selectedNote}</p>
          </div>
        )}
        <div className="page-info-links">
          <a href={`?files=${base}.jpg,${base}.streets.json`}>streets view</a>
          <a href={`?files=${base}.jpg,${base}.georef.json`}>georef view</a>
        </div>
      </div>
    );
  }

  if (pages.length === 0) {
    return (
      <div className="page-info-panel">
        <div className="page-info-header">
          <strong>Volume</strong>
        </div>
        <p className="page-info-hint">
          Select a volume to view its pages on the map.
        </p>
      </div>
    );
  }

  return (
    <div className="page-info-panel">
      <div className="page-info-header">
        <strong>{annotationName}</strong>
      </div>
      <dl>
        <dt>Pages</dt>
        <dd>{pages.length}</dd>
        {skipped.length > 0 && (
          <>
            <dt>Skipped</dt>
            <dd>{skipped.length}</dd>
          </>
        )}
        <dt>GCPs</dt>
        <dd>{pages.reduce((sum, page) => sum + page.gcps.length, 0)}</dd>
        <dt>Fits</dt>
        <dd>{fitSummary(pages)}</dd>
        <dt>Median scale</dt>
        <dd>
          {median(pages.map((p) => p.scalePixelsPerFoot)).toFixed(2)} px/ft
        </dd>
      </dl>
      <p className="page-info-hint">Click a page for details.</p>
    </div>
  );
}
