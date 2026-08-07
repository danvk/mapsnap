import type { ReactElement } from 'react';
import type { KeymapInfo } from '../../server/api';
import type { SkippedItem } from '../../server/iiifAnnotations';
import type { PageCompareStats } from '../iiif/compare';
import { hasFootprint, type PageGeo } from '../iiif/pages';

/**
 * One page view, offered both inline and as a standalone tab.
 *
 * Clicking the label opens it in place of the map, which keeps the volume, its
 * page list and the current selection on screen. The adjacent arrow is a real
 * link, so ctrl/cmd-click and "open in new tab" keep working exactly as before
 * -- that was the habit this replaces, not one to take away.
 */
function DebugViewLink({
  label,
  files,
  onOpen,
}: {
  label: string;
  files: string[];
  onOpen?: (files: string[], label: string) => void;
}): ReactElement {
  const href = `?files=${files.join(',')}`;
  return (
    <span className="debug-view-link">
      {onOpen ? (
        <button type="button" onClick={() => onOpen(files, label)}>
          {label}
        </button>
      ) : (
        <a href={href}>{label}</a>
      )}
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="debug-view-newtab"
        title={`Open ${label} in a new tab`}
        aria-label={`Open ${label} in a new tab`}
      >
        ↗
      </a>
    </span>
  );
}

/** A run's own artifact directory and the pages it saved sidecars for. */
export interface RunArtifacts {
  dir: string;
  stems: string[];
}

interface InfoPanelProps {
  /** All pages in the loaded annotation, or [] before one is loaded. */
  pages: PageGeo[];
  /** How many truth pages the run never georeferenced, for the volume summary. */
  missingCount: number;
  /** Items the server dropped while rewriting the annotation. */
  skipped: SkippedItem[];
  /** The loaded annotation file's name, for the summary header. */
  annotationName: string | null;
  selectedPage: PageGeo | null;
  /** Whether the selected page is a missing (un-fitted) truth page. */
  selectedMissing: boolean;
  /**
   * Failure kind of the selected page's failed-georef sidecar ("nofit"/"1gcp"/…),
   * or null when it has none; drives the georef-view link for a missing page.
   */
  selectedFailedGeorefType: string | null;
  /** Truth-compare stats for the selected page, when the volume has truth. */
  selectedStats: PageCompareStats | null;
  /** The selected page's note text, or null when it has none. */
  selectedNote: string | null;
  /** Whether the volume has adjacency data (adds an adjacency-view link). */
  hasAdjacency: boolean;
  /** The compare table's summary footer, shown in the no-selection view; "" if none. */
  compareFooter: string;
  /** The volume's key-map sheets, linked to their visualizations in the no-selection view. */
  keymaps: KeymapInfo[];
  /** Volume directory name, e.g. "brooklyn_ny_1906_vol_6". */
  volume: string;
  /**
   * Opens a page view inline, in place of the map. Absent in contexts with
   * nowhere to put it, in which case the labels stay plain links.
   */
  onOpenDebugView?: (files: string[], label: string) => void;
  /**
   * The run's own artifact directory and the page stems it holds sidecars for,
   * from GET /iiif-api/run-artifacts. Links prefer these over the top-level
   * sidecars, which belong to whatever ran most recently rather than to the
   * annotation on screen.
   */
  runArtifacts?: RunArtifacts;
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

// One key map's visualization links, e.g. "p0 (regions, georef)". The stem links to the
// key-map detection view; each present sidecar links to its own view via the `?files=` deep link.
function KeymapLinks({
  keymap,
  volume,
}: {
  keymap: KeymapInfo;
  volume: string;
}): ReactElement {
  const base = `data/${volume}/raw/${keymap.stem}`;
  const extras: ReactElement[] = [];
  if (keymap.hasRegions) {
    extras.push(
      <a key="regions" href={`?files=${base}.jpg,${base}.regions.panels.json`}>
        regions
      </a>,
    );
  }
  if (keymap.hasGeoref) {
    extras.push(
      <a key="georef" href={`?files=${base}.jpg,${base}.georef.json`}>
        georef
      </a>,
    );
  }
  return (
    <span>
      <a href={`?files=${base}.jpg,${base}.keymap.json`}>{keymap.stem}</a>
      {extras.length > 0 && (
        <>
          {' ('}
          {extras.map((link, i) => (
            <span key={i}>
              {i > 0 && ', '}
              {link}
            </span>
          ))}
          {')'}
        </>
      )}
    </span>
  );
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
/**
 * A similarity fit is exactly skew 0 / anisotropy 1, so any real deviation
 * comes from the annotation's own transform. A flat sheet photographed square
 * cannot shear, so past these tolerances the georeference is more likely bad
 * reference data than a real property of the map -- worth flagging in red
 * while auditing truth. Thresholds are loose enough that ordinary
 * polynomial-fit wobble stays quiet.
 */
const SKEW_WARN_DEGREES = 1.0;
const ANISOTROPY_WARN = 0.05;

/** The skew and anisotropy rows shared by the truth and generated stat blocks. */
function DistortionRows({ page }: { page: PageGeo }) {
  const skewBad = Math.abs(page.skewDegrees) > SKEW_WARN_DEGREES;
  const anisoBad = Math.abs(page.anisotropy - 1) > ANISOTROPY_WARN;
  return (
    <>
      <dt>Skew</dt>
      <dd className={skewBad ? 'gcp-stats-warning' : undefined}>
        {page.skewDegrees.toFixed(2)}°
      </dd>
      <dt>Anisotropy</dt>
      <dd className={anisoBad ? 'gcp-stats-warning' : undefined}>
        {page.anisotropy.toFixed(3)}
      </dd>
    </>
  );
}

export function InfoPanel(props: InfoPanelProps) {
  const {
    pages,
    missingCount,
    skipped,
    annotationName,
    selectedPage,
    selectedMissing,
    selectedFailedGeorefType,
    selectedStats,
    selectedNote,
    hasAdjacency,
    compareFooter,
    keymaps,
    volume,
    runArtifacts,
    onOpenDebugView,
    onClose,
  } = props;

  if (selectedPage) {
    // Prefer the run's own sidecar for this page; fall back to the volume root
    // when this run did not save one, and say so rather than linking silently
    // to a file that may have come from a different run.
    //
    // Only the sidecar moves. An artifact directory holds a run's json and txt
    // output, never page images -- those exist once, at the volume root -- so
    // the image and its sidecar cannot share a base path.
    const fromRun = !!runArtifacts?.stems.includes(selectedPage.stem);
    const imageBase = `data/${volume}/${selectedPage.stem}`;
    const sidecarBase = fromRun
      ? `${runArtifacts!.dir}/${selectedPage.stem}`
      : imageBase;

    // A not-georeferenced page has no `<stem>.georef.json`; it has whichever
    // failure sidecar the fit wrote, so link that instead of a missing file.
    const georefFiles = selectedMissing
      ? selectedFailedGeorefType
        ? [
            `${imageBase}.jpg`,
            `${sidecarBase}.georef-${selectedFailedGeorefType}.json`,
          ]
        : null
      : [`${imageBase}.jpg`, `${sidecarBase}.georef.json`];
    const debugViews: { label: string; files: string[] }[] = [
      {
        label: 'streets view',
        files: [`${imageBase}.jpg`, `${sidecarBase}.streets.json`],
      },
      ...(georefFiles
        ? [
            {
              label: selectedMissing
                ? `georef view (${selectedFailedGeorefType})`
                : 'georef view',
              files: georefFiles,
            },
          ]
        : []),
      ...(hasAdjacency
        ? [
            {
              label: 'adjacency view',
              // Adjacency is volume-wide, not per run.
              files: [`${imageBase}.jpg`, `data/${volume}/adjacency.json`],
            },
          ]
        : []),
    ];
    return (
      <div className="page-info-panel">
        <div className="page-info-header">
          <strong>{selectedPage.stem}</strong>
          <button type="button" onClick={onClose} title="Deselect page">
            ×
          </button>
        </div>
        <dl>
          {selectedMissing ? (
            <>
              <dt>Status</dt>
              <dd>not georeferenced</dd>
              {/* Everything below comes from the page's truth georeference. A volume
                  without truth knows only that the page exists and was not placed. */}
              {hasFootprint(selectedPage) && (
                <>
                  <dt>Truth scale</dt>
                  <dd>{selectedPage.scalePixelsPerFoot.toFixed(2)} px/ft</dd>
                  <dt>Truth rotation</dt>
                  <dd>{selectedPage.rotationDegrees.toFixed(1)}°</dd>
                  <DistortionRows page={selectedPage} />
                  <dt>Size</dt>
                  <dd>
                    {selectedPage.width} × {selectedPage.height} px
                  </dd>
                </>
              )}
            </>
          ) : (
            <>
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
              <DistortionRows page={selectedPage} />
              <dt>Size</dt>
              <dd>
                {selectedPage.width} × {selectedPage.height} px
              </dd>
              <dt>GCPs</dt>
              <dd>{selectedPage.gcps.length}</dd>
              <dt>Fit</dt>
              <dd>{selectedPage.transformationType}</dd>
            </>
          )}
        </dl>
        {selectedNote && (
          <div className="page-info-note">
            <span className="page-info-note-label">📓 Note</span>
            <p>{selectedNote}</p>
          </div>
        )}
        {runArtifacts?.dir && !fromRun && (
          <p className="page-info-fallback">
            This run saved no sidecar for this page; the links below point at
            the volume&rsquo;s top-level files, which may come from a different
            run.
          </p>
        )}
        {!runArtifacts?.dir && (
          <p className="page-info-fallback">
            This run has no artifact directory; the links below point at the
            volume&rsquo;s top-level files, which may come from a different run.
          </p>
        )}
        <div className="page-info-links">
          {debugViews.map(({ label, files }) => (
            <DebugViewLink
              key={label}
              label={label}
              files={files}
              onOpen={onOpenDebugView}
            />
          ))}
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
        {missingCount > 0 && (
          <>
            <dt>Missing</dt>
            <dd>{missingCount}</dd>
          </>
        )}
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
      {keymaps.length > 0 && (
        <div className="page-info-keymaps">
          Keymaps:{' '}
          {keymaps.map((keymap, i) => (
            <span key={keymap.stem}>
              {i > 0 && ', '}
              <KeymapLinks keymap={keymap} volume={volume} />
            </span>
          ))}
        </div>
      )}
      {compareFooter && (
        <pre className="page-info-compare-footer">{compareFooter}</pre>
      )}
      <p className="page-info-hint">Click a page for details.</p>
    </div>
  );
}
