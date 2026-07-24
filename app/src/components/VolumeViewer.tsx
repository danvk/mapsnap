import { useEffect, useMemo, useState } from 'react';
import type {
  GeorefAnnotationPage,
  SkippedItem,
  VolumeInfo,
} from '../../server/iiifAnnotations';
import {
  RMSE_BUCKET_COLORS,
  rmseBucket,
  statsByItemIndex,
  type PageCompareStats,
} from '../iiif/compare';
import type { ComparePageStats } from '../../server/compareTxt';
import type { AdjacencyData } from '../types';
import {
  fetchAdjacency,
  fetchCompare,
  fetchFailedGeorefs,
  fetchKeymaps,
  fetchRewrittenAnnotation,
  fetchVolumes,
} from '../iiif/api';
import type { KeymapInfo } from '../../server/api';
import { isTypingTarget } from '../keyboard';
import { fetchVolumeNotes } from '../notes/api';
import { adjacencyClaimFeatures } from '../iiif/adjacency';
import { missingTruthPages, pagesFromAnnotation } from '../iiif/pages';
import { InfoPanel } from './InfoPanel';
import { PageList } from './PageList';
import { VolumeMap } from './VolumeMap';

// Split a repo-root-relative annotation path like
// "data/brooklyn_ny_1906_vol_6/generated.iiif.json" into volume + file name.
function parseAnnotationPath(
  path: string | null,
): { volume: string; file: string } | null {
  const match = path?.match(/^data\/([^/]+)\/([^/]+)$/);
  return match ? { volume: match[1] ?? '', file: match[2] ?? '' } : null;
}

// Map viewport from the URL's `center=lng,lat` and `zoom=Z` params, or null when absent/invalid.
function parseViewport(
  params: URLSearchParams,
): { center: [number, number]; zoom: number } | null {
  const center = params.get('center')?.split(',').map(Number);
  const zoom = Number(params.get('zoom'));
  if (
    !center ||
    center.length !== 2 ||
    ![...center, zoom].every(Number.isFinite)
  )
    return null;
  return { center: [center[0]!, center[1]!], zoom };
}

// Merge updates into the current URL query and replace history (null value deletes a key).
function updateUrl(updates: Record<string, string | null>): void {
  const params = new URLSearchParams(window.location.search);
  for (const [key, value] of Object.entries(updates)) {
    if (value === null) params.delete(key);
    else params.set(key, value);
  }
  history.replaceState(null, '', `?${params}`);
}

/**
 * Full-volume IIIF viewer: pick a volume and one of its georeference
 * annotation files, and every georeferenced page is shown warped and clipped
 * on the map, with images served by the local IIIF server.
 */
export function VolumeViewer() {
  const [volumes, setVolumes] = useState<VolumeInfo[] | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get('iiif'),
  );
  const [annotation, setAnnotation] = useState<unknown>(null);
  const [skipped, setSkipped] = useState<SkippedItem[]>([]);
  const [loadResult, setLoadResult] = useState<{
    loaded: number;
    failed: number;
  } | null>(null);
  const [opacity, setOpacity] = useState(100);
  // View state seeded from the URL so a shared/reloaded link restores the same view, and so
  // switching annotation files within a volume keeps the selection, checkboxes, and viewport.
  const initialParams = new URLSearchParams(window.location.search);
  const [colorByRmse, setColorByRmse] = useState(
    () => initialParams.get('rmse') === '1',
  );
  const [showMissing, setShowMissing] = useState(
    () => initialParams.get('missing') === '1',
  );
  const [showAdjacency, setShowAdjacency] = useState(
    () => initialParams.get('adj') === '1',
  );
  const [error, setError] = useState<string | null>(null);
  // Selection is tracked by page stem (stable across annotation files, unlike the item index).
  const [selectedStem, setSelectedStem] = useState<string | null>(() =>
    initialParams.get('page'),
  );
  const [initialViewport] = useState(() => parseViewport(initialParams));
  const [truthAnnotation, setTruthAnnotation] = useState<unknown>(null);
  // Paired-page error stats from the annotation's `mapsnap compare` sidecar, or null when
  // there is no sidecar (no truth comparison for this annotation).
  const [compareRows, setCompareRows] = useState<ComparePageStats[] | null>(
    null,
  );
  // The compare table's summary footer ("N/M pages georeferenced", RMSE stats), or "" if none.
  const [compareFooter, setCompareFooter] = useState<string>('');
  // The selected volume's adjacency.json (per-page claims + mutual graph), or null when absent.
  const [adjacencyData, setAdjacencyData] = useState<AdjacencyData | null>(
    null,
  );
  // Page key → note text for the selected volume (markers + tooltip).
  const [notes, setNotes] = useState<Map<string, string>>(new Map());
  // Page stem → failed-georef kind ("nofit"/"1gcp"/…) for the selected volume.
  const [failedGeorefs, setFailedGeorefs] = useState<Map<string, string>>(
    new Map(),
  );
  // The selected volume's key-map sheets (raw/*.keymap.json), for the info-panel links.
  const [keymaps, setKeymaps] = useState<KeymapInfo[]>([]);

  useEffect(() => {
    fetchVolumes()
      .then((resp) => setVolumes(resp.volumes))
      .catch((err) => setError(String(err)));
  }, []);

  // Load the selected annotation and keep the ?iiif= deep link in sync. The selection is not
  // reset here: it is keyed by stem, so it carries across annotation files within a volume and
  // simply resolves to nothing when the stem is absent from a newly chosen volume.
  useEffect(() => {
    if (!selectedPath) return;
    let cancelled = false;
    setError(null);
    setLoadResult(null);
    fetchRewrittenAnnotation(selectedPath)
      .then((resp) => {
        if (cancelled) return;
        setAnnotation(resp.annotation);
        setSkipped(resp.skipped);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    updateUrl({ view: 'iiif', iiif: selectedPath });

    // Per-page truth error and summary footer from this annotation's `mapsnap compare` sidecar.
    setCompareRows(null);
    setCompareFooter('');
    fetchCompare(selectedPath)
      .then(({ pages, footer }) => {
        if (cancelled) return;
        setCompareRows(pages);
        setCompareFooter(footer);
      })
      .catch(() => {
        if (!cancelled) setCompareRows([]);
      });

    // Truth annotation, rewritten into the same local pixel frame, for the missing-page
    // footprints. Skipped when viewing the truth itself.
    setTruthAnnotation(null);
    const parsed = parseAnnotationPath(selectedPath);
    if (parsed && parsed.file !== 'main.iiif.json') {
      fetchRewrittenAnnotation(`data/${parsed.volume}/main.iiif.json`)
        .then((resp) => {
          if (!cancelled) setTruthAnnotation(resp.annotation);
        })
        .catch(() => {
          // No truth data for this volume; the list simply has no RMSE column.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [selectedPath]);

  const selection = parseAnnotationPath(selectedPath);
  const selectedVolume = volumes?.find((v) => v.name === selection?.volume);

  // Load the selected volume's page notes, failed-georef sidecars, and adjacency data: the
  // notes drive the list markers/tooltip, the failed-georefs the missing-page links, the
  // adjacency the claim overlay.
  const volumeName = selection?.volume;
  useEffect(() => {
    if (!volumeName) {
      setNotes(new Map());
      setFailedGeorefs(new Map());
      setAdjacencyData(null);
      setKeymaps([]);
      return;
    }
    let cancelled = false;
    fetchKeymaps(volumeName)
      .then((list) => {
        if (!cancelled) setKeymaps(list);
      })
      .catch(() => {
        if (!cancelled) setKeymaps([]);
      });
    fetchVolumeNotes(volumeName)
      .then((map) => {
        if (!cancelled) setNotes(map);
      })
      .catch(() => {
        if (!cancelled) setNotes(new Map());
      });
    fetchFailedGeorefs(volumeName)
      .then((map) => {
        if (!cancelled) setFailedGeorefs(map);
      })
      .catch(() => {
        if (!cancelled) setFailedGeorefs(new Map());
      });
    setAdjacencyData(null);
    fetchAdjacency(volumeName)
      .then((data) => {
        if (!cancelled) setAdjacencyData(data);
      })
      .catch(() => {
        if (!cancelled) setAdjacencyData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [volumeName]);

  // Cycle warped-image opacity through 0/50/100% on the 'p' key, matching the
  // georef view (skipped while the user is typing).
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'p' || isTypingTarget(e.target)) return;
      const steps = [0, 50, 100];
      setOpacity(
        (prev) => steps[(steps.indexOf(prev) + 1) % steps.length] ?? 0,
      );
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, []);

  const pages = useMemo(
    () =>
      annotation ? pagesFromAnnotation(annotation as GeorefAnnotationPage) : [],
    [annotation],
  );
  const truthPages = useMemo(
    () =>
      truthAnnotation
        ? pagesFromAnnotation(truthAnnotation as GeorefAnnotationPage)
        : null,
    [truthAnnotation],
  );
  // Per-page compare stats keyed by itemIndex, from the sidecar rows. Null when the annotation
  // has no compare sidecar; empty rows also read as "no comparison".
  const truthStats: Map<number, PageCompareStats> | null = useMemo(
    () =>
      compareRows && compareRows.length > 0
        ? statsByItemIndex(compareRows, pages)
        : null,
    [compareRows, pages],
  );
  // Truth pages the run never georeferenced, shown as "missing" rows/footprints.
  const missingPages = useMemo(
    () => (truthPages ? missingTruthPages(pages, truthPages) : []),
    [pages, truthPages],
  );

  // Adjacency claim boxes, mapped into geo through each page's georeference: the fitted pages,
  // plus the missing pages (via their truth georef) when those are being shown.
  const adjacencyClaims = useMemo(() => {
    if (!adjacencyData) return [];
    const withGeoref = showMissing ? [...pages, ...missingPages] : pages;
    return adjacencyClaimFeatures(adjacencyData, withGeoref);
  }, [adjacencyData, pages, missingPages, showMissing]);

  const pageColors: Map<number, string> | null = useMemo(() => {
    if (!colorByRmse || !truthStats) return null;
    const colors = new Map<number, string>();
    for (const [itemIndex, stats] of truthStats) {
      colors.set(itemIndex, RMSE_BUCKET_COLORS[rmseBucket(stats.rmseFt)]);
    }
    return colors;
  }, [colorByRmse, truthStats]);

  // The selected page (fitted or missing) resolved from its stem, or null when nothing is
  // selected or the stem is absent from the current annotation. A missing page carries a
  // negative synthetic id, so it is found in missingPages; the info panel renders it differently.
  const selectedPage =
    selectedStem === null
      ? null
      : (pages.find((p) => p.stem === selectedStem) ??
        missingPages.find((p) => p.stem === selectedStem) ??
        null);
  const selectedItemIndex = selectedPage?.itemIndex ?? null;
  const selectedIsMissing =
    selectedPage !== null &&
    selectedItemIndex !== null &&
    selectedItemIndex < 0;

  // Selection is set by page stem so it survives an annotation-file switch (item ids differ).
  function handleSelectPage(itemIndex: number | null): void {
    if (itemIndex === null) {
      setSelectedStem(null);
      return;
    }
    const page =
      pages.find((p) => p.itemIndex === itemIndex) ??
      missingPages.find((p) => p.itemIndex === itemIndex);
    setSelectedStem(page?.stem ?? null);
  }

  // Mirror the selection and toggle state into the URL (the map writes the viewport itself).
  useEffect(() => {
    updateUrl({
      page: selectedStem,
      rmse: colorByRmse ? '1' : null,
      missing: showMissing ? '1' : null,
      adj: showAdjacency ? '1' : null,
    });
  }, [selectedStem, colorByRmse, showMissing, showAdjacency]);

  function selectVolume(name: string): void {
    const volume = volumes?.find((v) => v.name === name);
    const newest = volume?.annotations[0];
    if (volume && newest) setSelectedPath(`data/${volume.name}/${newest.name}`);
  }

  let status: string;
  if (error) {
    status = error;
  } else if (loadResult) {
    const parts = [`${loadResult.loaded} pages shown`];
    if (loadResult.failed > 0) parts.push(`${loadResult.failed} failed`);
    if (skipped.length > 0) parts.push(`${skipped.length} skipped`);
    status = parts.join(', ');
  } else if (selectedPath) {
    status = 'loading…';
  } else {
    status = 'Select a volume to view it on the map.';
  }

  return (
    <div className="volume-viewer">
      <div className="iiif-controls">
        <a href=".">← debugger</a>
        <select
          value={selection?.volume ?? ''}
          onChange={(e) => selectVolume(e.target.value)}
        >
          <option value="" disabled>
            Select a volume…
          </option>
          {(volumes ?? []).map((volume) => (
            <option key={volume.name} value={volume.name}>
              {volume.name} ({volume.pageCount} pages)
            </option>
          ))}
        </select>
        <select
          value={selection?.file ?? ''}
          onChange={(e) =>
            setSelectedPath(`data/${selection?.volume}/${e.target.value}`)
          }
          disabled={!selectedVolume}
        >
          {(selectedVolume?.annotations ?? []).map((file) => (
            <option key={file.name} value={file.name}>
              {file.name} ({file.itemCount})
            </option>
          ))}
        </select>
        {truthStats && (
          <label className="rmse-color-control">
            <input
              type="checkbox"
              checked={colorByRmse}
              onChange={(e) => setColorByRmse(e.target.checked)}
            />
            Color by RMSE
          </label>
        )}
        {missingPages.length > 0 && (
          <label className="rmse-color-control">
            <input
              type="checkbox"
              checked={showMissing}
              onChange={(e) => setShowMissing(e.target.checked)}
            />
            Show missing pages
          </label>
        )}
        {adjacencyData && (
          <label className="rmse-color-control">
            <input
              type="checkbox"
              checked={showAdjacency}
              onChange={(e) => setShowAdjacency(e.target.checked)}
            />
            Show adjacency
          </label>
        )}
        <div className="opacity-control">
          <label htmlFor="iiif-opacity-slider">Opacity</label>
          <input
            type="range"
            id="iiif-opacity-slider"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </div>
        <span className="iiif-status">{status}</span>
      </div>
      <div className="volume-viewer-body">
        <PageList
          pages={pages}
          missingPages={missingPages}
          stats={truthStats}
          notes={notes}
          selectedItemIndex={selectedItemIndex}
          onSelectPage={handleSelectPage}
        />
        <VolumeMap
          annotation={annotation}
          pages={pages}
          missingPages={missingPages}
          truthPages={truthPages ?? []}
          showMissing={showMissing}
          selectedItemIndex={selectedItemIndex}
          onSelectPage={handleSelectPage}
          opacity={opacity / 100}
          awaitingView={!!selectedPath && !error}
          pageColors={pageColors}
          adjacencyClaims={showAdjacency ? adjacencyClaims : []}
          selectedStem={selectedPage?.stem ?? null}
          initialViewport={initialViewport}
          fitVolumeKey={volumeName ?? null}
          onViewportChange={(center, zoom) =>
            updateUrl({
              center: `${center[0].toFixed(5)},${center[1].toFixed(5)}`,
              zoom: zoom.toFixed(2),
            })
          }
          onLoadResult={setLoadResult}
        />
        <InfoPanel
          pages={pages}
          missingCount={missingPages.length}
          skipped={skipped}
          annotationName={selection?.file ?? null}
          selectedPage={selectedPage}
          selectedMissing={selectedIsMissing}
          selectedFailedGeorefType={
            selectedPage ? (failedGeorefs.get(selectedPage.stem) ?? null) : null
          }
          selectedStats={
            selectedItemIndex === null
              ? null
              : (truthStats?.get(selectedItemIndex) ?? null)
          }
          selectedNote={
            selectedPage ? (notes.get(selectedPage.stem) ?? null) : null
          }
          hasAdjacency={adjacencyData !== null}
          compareFooter={compareFooter}
          keymaps={keymaps}
          volume={selection?.volume ?? ''}
          onClose={() => setSelectedStem(null)}
        />
      </div>
    </div>
  );
}
