import { useEffect, useMemo, useState } from 'react';
import {
  keymapUnderlays,
  underlayImageFromParam,
  underlayImageParam,
  type KeymapUnderlayImage,
} from '../iiif/underlay';
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
  fetchOsmRelation,
  fetchAdjacency,
  fetchCompare,
  fetchKeymaps,
  fetchRewrittenAnnotation,
  fetchRunArtifacts,
  fetchVolumePageFiles,
  fetchVolumes,
} from '../iiif/api';
import type { KeymapInfo } from '../../server/api';
import { isTypingTarget } from '../keyboard';
import { fetchVolumeNotes } from '../notes/api';
import { adjacencyClaimFeatures } from '../iiif/adjacency';
import {
  missingTruthPages,
  pagesFromAnnotation,
  unfittedPages,
} from '../iiif/pages';
import { DebugView } from '../App';
import { SnapPanel } from './SnapPanel';
import { parseSnapRecords, poseCorners, rankedCandidates } from '../snap';
import type { SnapRecord } from '../snap';
import { InfoPanel, type RunArtifacts } from './InfoPanel';
import { PageList } from './PageList';
import { VolumeMap } from './VolumeMap';
import { parseAnnotationPath } from '../iiif/volumePath';

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
  // Default ON: a page outside the downloaded relation cannot be fit at all,
  // and that is invisible without the ring.
  const [showOsmRelation, setShowOsmRelation] = useState(
    () => initialParams.get('osm') !== '0',
  );
  const [isolateSelected, setIsolateSelected] = useState(
    () => initialParams.get('only') === '1',
  );
  // The key-map underlay (#211): which image, and how strongly it shows. The
  // opacity is independent of the pages' so the two can be cross-faded.
  const [underlayImage, setUnderlayImage] = useState<KeymapUnderlayImage>(() =>
    underlayImageFromParam(initialParams.get('underlay')),
  );
  const [keymapOpacity, setKeymapOpacity] = useState(() => {
    const value = Number(initialParams.get('keymap'));
    return Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 0;
  });
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
  // Truth page keys the compare table marks `(no fit)` — the misses it counts, which is
  // what the missing rows show, so the two never disagree.
  const [compareMissing, setCompareMissing] = useState<string[]>([]);
  // The compare table's summary footer ("N/M pages georeferenced", RMSE stats), or "" if none.
  const [compareFooter, setCompareFooter] = useState<string>('');
  // The annotation's own artifact directory, so page links point at the files
  // that produced it rather than at whatever the last run left at the top level.
  const [runArtifacts, setRunArtifacts] = useState<RunArtifacts | undefined>();
  // A page view opened in place of the map. Null shows the map. The volume,
  // page list and selection are deliberately untouched by this, so closing the
  // view returns to exactly what was on screen before.
  const [debugView, setDebugView] = useState<{
    files: string[];
    label: string;
  } | null>(null);
  // Snap-channel records for the volume (candidates.jsonl, #325), keyed by
  // page stem; empty when the volume has no snap artifacts.
  const [snapRecords, setSnapRecords] = useState<Map<string, SnapRecord>>(
    new Map(),
  );
  // Whether the snap panel is open for the selected page, and which pose row
  // is highlighted (index into rankedCandidates, -1 for the incumbent).
  const [snapOpen, setSnapOpen] = useState(false);
  const [snapSelected, setSnapSelected] = useState<number | null>(null);
  useEffect(() => {
    setSnapOpen(false);
    setSnapSelected(null);
  }, [selectedStem]);
  // The selected volume's adjacency.json (per-page claims + mutual graph), or null when absent.
  const [osmRelation, setOsmRelation] = useState<{
    id: string;
    name: string | null;
    bufferM: number | null;
    ways: [number, number][][];
  } | null>(null);
  const [adjacencyData, setAdjacencyData] = useState<AdjacencyData | null>(
    null,
  );
  // Page key → note text for the selected volume (markers + tooltip).
  const [notes, setNotes] = useState<Map<string, string>>(new Map());
  // Page stem → its georef sidecars, for the selected volume.
  const [georefSidecars, setGeorefSidecars] = useState<Map<string, string[]>>(
    new Map(),
  );
  // Every page-image stem in the selected volume; the un-fit list for a volume with
  // no truth annotation is these minus the ones the run placed.
  const [volumeStems, setVolumeStems] = useState<string[]>([]);
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
    setRunArtifacts(undefined);
    fetchCompare(selectedPath)
      .then(({ pages, missing, footer }) => {
        if (cancelled) return;
        setCompareRows(pages);
        setCompareMissing(missing ?? []);
        setCompareFooter(footer);
      })
      .catch(() => {
        if (cancelled) return;
        setCompareRows([]);
        setCompareMissing([]);
      });

    fetchRunArtifacts(selectedPath)
      .then(({ dir, stems }) => {
        if (cancelled) return;
        setRunArtifacts(dir ? { dir, stems } : undefined);
      })
      .catch(() => {
        // An older server without this endpoint degrades to the top-level
        // links, which is what the viewer did before this existed.
        if (!cancelled) setRunArtifacts(undefined);
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

  // Load the selected volume's page notes, georef sidecars, and adjacency data: the
  // notes drive the list markers/tooltip, the sidecars the per-page georef links, the
  // adjacency the claim overlay.
  const volumeName = selection?.volume;
  useEffect(() => {
    if (!volumeName) {
      setNotes(new Map());
      setGeorefSidecars(new Map());
      setAdjacencyData(null);
      setOsmRelation(null);
      setKeymaps([]);
      return;
    }
    let cancelled = false;
    fetchOsmRelation(volumeName)
      .then((relation) => {
        if (!cancelled) setOsmRelation(relation);
      })
      .catch(() => {
        if (!cancelled) setOsmRelation(null);
      });
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
    fetchVolumePageFiles(volumeName)
      .then(({ stems, georefs }) => {
        if (cancelled) return;
        setGeorefSidecars(georefs);
        setVolumeStems(stems);
      })
      .catch(() => {
        if (cancelled) return;
        setGeorefSidecars(new Map());
        setVolumeStems([]);
      });
    setSnapRecords(new Map());
    setSnapOpen(false);
    setSnapSelected(null);
    // The snap channel's per-page record (#325). JSONL, so fetched as text; a
    // volume without snap artifacts 404s into the SPA fallback, which fails
    // the parse harmlessly (no rows -> empty map).
    fetch(`/data/${volumeName}/artifacts/osm_snap/candidates.jsonl`)
      .then((r) => (r.ok ? r.text() : ''))
      .then((text) => {
        if (!cancelled) setSnapRecords(parseSnapRecords(text));
      })
      .catch(() => {
        if (!cancelled) setSnapRecords(new Map());
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

  // The same 0/50/100% cycle for the key-map underlay, on 'k'.
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'k' || isTypingTarget(e.target)) return;
      const steps = [0, 50, 100];
      setKeymapOpacity(
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
  // Pages the run never georeferenced, shown as "missing" rows (and footprints where
  // truth gives one). With truth the misses come from the comparison, which decides
  // them per truth page; without it there is nothing to compare against, so they are
  // the volume's page images minus the ones the annotation placed.
  const missingPages = useMemo(
    () =>
      truthPages
        ? missingTruthPages(truthPages, compareMissing)
        : unfittedPages(pages, volumeStems),
    [truthPages, compareMissing, pages, volumeStems],
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
  // The selected page's snap record and, when a pose row is highlighted, the
  // P(road) overlay for it (image at the pose's own corner coordinates).
  const snapRecord =
    selectedStem !== null ? (snapRecords.get(selectedStem) ?? null) : null;
  const snapOverlay = useMemo(() => {
    if (!snapOpen || !snapRecord || snapSelected === null || !volumeName)
      return null;
    const pose =
      snapSelected === -1
        ? snapRecord.incumbent
        : snapSelected === -2
          ? snapRecord.truth_pose
          : rankedCandidates(snapRecord)[snapSelected];
    if (!pose?.world_affine) return null;
    return {
      url: `/data/${volumeName}/artifacts/edge_join/roadprob/${snapRecord.target}.png`,
      corners: poseCorners(
        pose.world_affine,
        snapRecord.width,
        snapRecord.height,
      ),
    };
  }, [snapOpen, snapRecord, snapSelected, volumeName]);
  // Nothing is fetched until the underlay is actually turned up.
  const underlays = useMemo(
    () =>
      volumeName && keymapOpacity > 0
        ? keymapUnderlays(volumeName, keymaps, underlayImage)
        : [],
    [volumeName, keymaps, underlayImage, keymapOpacity],
  );
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
      osm: showOsmRelation ? null : '0',
      only: isolateSelected ? '1' : null,
      underlay: underlayImageParam(underlayImage),
      keymap: keymapOpacity > 0 ? String(keymapOpacity) : null,
    });
  }, [
    selectedStem,
    colorByRmse,
    showMissing,
    showAdjacency,
    showOsmRelation,
    isolateSelected,
    underlayImage,
    keymapOpacity,
  ]);

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
        <label
          className="rmse-color-control"
          title={
            selectedStem
              ? `Show only ${selectedStem}`
              : 'Select a page to isolate it'
          }
        >
          <input
            type="checkbox"
            checked={isolateSelected}
            disabled={!selectedStem}
            onChange={(e) => setIsolateSelected(e.target.checked)}
          />
          Isolate selected
        </label>
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
        {osmRelation && (
          <label
            className="rmse-color-control"
            title={`Streets were downloaded from OSM ${osmRelation.id}${
              osmRelation.name ? ` (${osmRelation.name})` : ''
            }${
              osmRelation.bufferM
                ? `, buffered by ${osmRelation.bufferM} m — the ring is the area actually downloaded, not the administrative line`
                : ' (no buffer — the ring is the administrative boundary itself)'
            }. Pages crossing this ring cover ground whose streets are missing from the vocabulary.`}
          >
            <input
              type="checkbox"
              checked={showOsmRelation}
              onChange={(e) => setShowOsmRelation(e.target.checked)}
            />
            {osmRelation.bufferM
              ? `OSM boundary +${
                  osmRelation.bufferM >= 1000
                    ? `${osmRelation.bufferM / 1000} km`
                    : `${osmRelation.bufferM} m`
                }`
              : 'OSM boundary'}
          </label>
        )}
        {keymaps.some((keymap) => keymap.hasGeoref && keymap.hasRoadprob) && (
          <label
            className="rmse-color-control"
            title="Show the key map's P(road) map (raw/<stem>.roadprob.png) in place of the sheet: what a key-map snap would match against."
          >
            <input
              type="checkbox"
              checked={underlayImage === 'roadprob'}
              onChange={(e) =>
                setUnderlayImage(e.target.checked ? 'roadprob' : 'sheet')
              }
            />
            as P(road)
          </label>
        )}
        {keymaps.some((keymap) => keymap.hasGeoref) && (
          <div
            className="opacity-control"
            title="Key-map underlay opacity, independent of the pages' (#211). Press k to cycle 0/50/100%."
          >
            <label htmlFor="iiif-keymap-opacity-slider">Key map</label>
            <input
              type="range"
              id="iiif-keymap-opacity-slider"
              min={0}
              max={100}
              value={keymapOpacity}
              onChange={(e) => setKeymapOpacity(Number(e.target.value))}
            />
          </div>
        )}
        <div
          className="opacity-control"
          title="Page opacity. Press p to cycle 0/50/100%."
        >
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
        {/* The map stays mounted while a debug view is open, hidden rather than
            unmounted. Unmounting destroys the maplibre instance, and remounting
            re-runs its initial fitBounds and refetches every tile -- a visible
            jump and reload on what should be a return to where you were. Hiding
            it keeps its layout box, so nothing resizes either. */}
        <div className="volume-map-slot" aria-hidden={debugView !== null}>
          <VolumeMap
            snapOverlay={snapOverlay}
            keymapUnderlays={underlays}
            keymapOpacity={keymapOpacity / 100}
            annotation={annotation}
            pages={pages}
            missingPages={missingPages}
            truthPages={truthPages ?? []}
            showMissing={showMissing}
            isolateSelected={isolateSelected}
            selectedItemIndex={selectedItemIndex}
            onSelectPage={handleSelectPage}
            opacity={opacity / 100}
            awaitingView={!!selectedPath && !error}
            pageColors={pageColors}
            adjacencyClaims={showAdjacency ? adjacencyClaims : []}
            osmRelationWays={
              showOsmRelation && osmRelation ? osmRelation.ways : null
            }
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
          {snapOpen && snapRecord && (
            <div className="snap-panel-slot">
              <SnapPanel
                record={snapRecord}
                selected={snapSelected}
                onSelect={setSnapSelected}
                onClose={() => setSnapOpen(false)}
              />
            </div>
          )}
          {debugView && (
            <div className="volume-viewer-debug">
              <DebugView
                key={debugView.files.join(',')}
                files={debugView.files}
                onClose={() => setDebugView(null)}
              />
            </div>
          )}
        </div>
        <InfoPanel
          onOpenDebugView={(files, label) => setDebugView({ files, label })}
          onOpenSnapView={snapRecord ? () => setSnapOpen(true) : undefined}
          runArtifacts={runArtifacts}
          pages={pages}
          missingCount={missingPages.length}
          skipped={skipped}
          annotationName={selection?.file ?? null}
          selectedPage={selectedPage}
          selectedMissing={selectedIsMissing}
          selectedGeorefFiles={
            selectedPage ? (georefSidecars.get(selectedPage.stem) ?? []) : []
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
          oimSlug={selectedVolume?.oimSlug ?? null}
          keymaps={keymaps}
          volume={selection?.volume ?? ''}
          onClose={() => setSelectedStem(null)}
        />
      </div>
    </div>
  );
}
