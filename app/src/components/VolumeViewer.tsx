import { useEffect, useMemo, useState } from 'react';
import type {
  GeorefAnnotationPage,
  SkippedItem,
  VolumeInfo,
} from '../../server/iiifAnnotations';
import {
  RMSE_BUCKET_COLORS,
  compareToTruth,
  rmseBucket,
  type PageCompareStats,
} from '../iiif/compare';
import {
  fetchFailedGeorefs,
  fetchRewrittenAnnotation,
  fetchVolumes,
} from '../iiif/api';
import { isTypingTarget } from '../keyboard';
import { fetchVolumeNotes } from '../notes/api';
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
  const [colorByRmse, setColorByRmse] = useState(false);
  const [showMissing, setShowMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedItemIndex, setSelectedItemIndex] = useState<number | null>(
    null,
  );
  const [truthAnnotation, setTruthAnnotation] = useState<unknown>(null);
  // Page key → note text for the selected volume (markers + tooltip).
  const [notes, setNotes] = useState<Map<string, string>>(new Map());
  // Page stem → failed-georef kind ("nofit"/"1gcp"/…) for the selected volume.
  const [failedGeorefs, setFailedGeorefs] = useState<Map<string, string>>(
    new Map(),
  );

  useEffect(() => {
    fetchVolumes()
      .then((resp) => setVolumes(resp.volumes))
      .catch((err) => setError(String(err)));
  }, []);

  // Load the selected annotation and keep the ?iiif= deep link in sync.
  useEffect(() => {
    if (!selectedPath) return;
    let cancelled = false;
    setError(null);
    setLoadResult(null);
    setSelectedItemIndex(null);
    fetchRewrittenAnnotation(selectedPath)
      .then((resp) => {
        if (cancelled) return;
        setAnnotation(resp.annotation);
        setSkipped(resp.skipped);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    const params = new URLSearchParams(window.location.search);
    params.set('view', 'iiif');
    params.set('iiif', selectedPath);
    history.replaceState(null, '', `?${params}`);

    // Truth for the compare column: the volume's main.iiif.json, rewritten
    // into the same local pixel frame. Skipped when viewing the truth itself.
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

  // Load the selected volume's page notes and failed-georef sidecars: the notes
  // drive the list markers/tooltip, the failed-georefs the missing-page links.
  const volumeName = selection?.volume;
  useEffect(() => {
    if (!volumeName) {
      setNotes(new Map());
      setFailedGeorefs(new Map());
      return;
    }
    let cancelled = false;
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
  const truthStats: Map<number, PageCompareStats | null> | null = useMemo(
    () => (truthPages ? compareToTruth(pages, truthPages) : null),
    [pages, truthPages],
  );
  // Truth pages the run never georeferenced, shown as "missing" rows/footprints.
  const missingPages = useMemo(
    () => (truthPages ? missingTruthPages(pages, truthPages) : []),
    [pages, truthPages],
  );

  const pageColors: Map<number, string> | null = useMemo(() => {
    if (!colorByRmse || !truthStats) return null;
    const colors = new Map<number, string>();
    for (const [itemIndex, stats] of truthStats) {
      if (stats) {
        colors.set(itemIndex, RMSE_BUCKET_COLORS[rmseBucket(stats.rmseFt)]);
      }
    }
    return colors;
  }, [colorByRmse, truthStats]);

  // A missing page carries a negative synthetic id, so it is found in
  // missingPages, not the fitted pages; the info panel renders it differently.
  const selectedPage =
    selectedItemIndex === null
      ? null
      : (pages.find((p) => p.itemIndex === selectedItemIndex) ??
        missingPages.find((p) => p.itemIndex === selectedItemIndex) ??
        null);
  const selectedIsMissing =
    selectedPage !== null &&
    selectedItemIndex !== null &&
    selectedItemIndex < 0;

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
          onSelectPage={setSelectedItemIndex}
        />
        <VolumeMap
          annotation={annotation}
          pages={pages}
          missingPages={missingPages}
          showMissing={showMissing}
          selectedItemIndex={selectedItemIndex}
          onSelectPage={setSelectedItemIndex}
          opacity={opacity / 100}
          awaitingView={!!selectedPath && !error}
          pageColors={pageColors}
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
            selectedPage
              ? (failedGeorefs.get(selectedPage.pageKey) ?? null)
              : null
          }
          selectedStats={
            selectedItemIndex === null
              ? null
              : (truthStats?.get(selectedItemIndex) ?? null)
          }
          selectedNote={
            selectedPage ? (notes.get(selectedPage.pageKey) ?? null) : null
          }
          volume={selection?.volume ?? ''}
          onClose={() => setSelectedItemIndex(null)}
        />
      </div>
    </div>
  );
}
