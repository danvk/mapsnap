import { useEffect, useMemo, useRef, useState } from 'react';
import './styles.css';
import type {
  AdjacencyData,
  Box,
  Corners,
  GcpPairResult,
  GeorefData,
  IntersectionPoint,
  KeymapLocation,
  PanelPolygon,
  Street,
  Detection,
} from './types';
import {
  computeCorners,
  directionThroughCorners,
  projectThroughCorners,
} from './geometry';
import {
  detectionFromAdjacency,
  filterDetections,
  type DetectionFilters,
  type IndexedDetection,
} from './detections';
import { pageStem, parseDroppedJson } from './fileLoading';
import { ImageColumn, type Mode } from './components/ImageColumn';
import { MapView } from './components/MapView';
import { GcpControls, type GcpFitStats } from './components/GcpControls';
import { AdjacencyTable } from './components/AdjacencyTable';
import { DetectionsTable } from './components/DetectionsTable';
import { PanelsTable } from './components/PanelsTable';
import { BoxControls } from './components/BoxControls';
import { VolumeViewer } from './components/VolumeViewer';
import { NoteButton } from './components/NoteButton';
import { noteContextFromFiles, type NoteContext } from './notes/api';
import { isTypingTarget } from './keyboard';
import { loadImage } from './loadImage';

// The seed pair the pipeline chose: the two intersections flagged `initial`.
function initialPairFrom(
  intersections: IntersectionPoint[],
): [number, number] | null {
  const idx = intersections
    .map((ix, i) => (ix.initial ? i : -1))
    .filter((i) => i >= 0);
  return idx.length === 2 ? [idx[0], idx[1]] : null;
}

// The highest-scoring non-degenerate seed pair, used as the default view for a nofit debug
// file where the pipeline selected no pair. Picks the max of precomputed scores (no re-scoring);
// returns null when there are no scored pairs.
function bestPairFrom(pairs: GcpPairResult[] | null): [number, number] | null {
  if (!pairs) return null;
  let best: GcpPairResult | null = null;
  for (const pair of pairs) {
    if (pair.degenerate || pair.score === undefined) continue;
    if (best === null || pair.score > best.score!) best = pair;
  }
  return best ? [best.a, best.b] : null;
}

// Find the recorded fit for an unordered seed pair, or null if it wasn't scored.
function findPairRecord(
  pairs: GcpPairResult[] | null,
  pair: [number, number] | null,
): GcpPairResult | null {
  if (!pairs || !pair) return null;
  const [a, b] = pair;
  return (
    pairs.find((p) => (p.a === a && p.b === b) || (p.a === b && p.b === a)) ??
    null
  );
}

// Whether a URL/path points at an image we can load (matched by extension).
function isImageUrl(url: string): boolean {
  return /\.(jpe?g|png|webp|gif|tiff?)(\?.*)?$/i.test(url);
}

// Resolve a `?files=` entry to a fetchable URL, leaving absolute URLs/paths as
// given and otherwise resolving relative to the app's base (e.g.
// `data/streets.json` -> `/mapsnap/data/streets.json`, served by the dev server).
function resolveDataUrl(file: string): string {
  if (/^https?:\/\//.test(file) || file.startsWith('/')) return file;
  return import.meta.env.BASE_URL.replace(/\/$/, '') + '/' + file;
}
/**
 * Debug API exposed on `window.mapsnap` so data can be injected without the UI
 * (e.g. from the browser console or automated tests). `loadJson` accepts either
 * georef JSON or a streets.json detection list; `setImage` points the viewer at
 * an image URL.
 */
export interface MapsnapDebugApi {
  loadJson: (text: string) => void;
  setImage: (url: string) => Promise<void>;
}

declare global {
  interface Window {
    mapsnap: MapsnapDebugApi;
  }
}

/**
 * Top-level debugger app with several modes: georef (map + overlays), streets
 * (text detections), panels (page-split polygons), and boxes (raw CRAFT
 * detection boxes per rotation).
 */
export function App() {
  const [mode, setMode] = useState<Mode>(() =>
    new URLSearchParams(window.location.search).get('view') === 'iiif'
      ? 'iiif'
      : 'georef',
  );
  const [streets, setStreets] = useState<Street[]>([]);
  const [intersections, setIntersections] = useState<IntersectionPoint[]>([]);
  const [keymap, setKeymap] = useState<KeymapLocation | null>(null);
  const [truth, setTruth] = useState<[number, number][][] | null>(null);
  const [gcpPairs, setGcpPairs] = useState<GcpPairResult[] | null>(null);
  const [selectedPair, setSelectedPair] = useState<[number, number] | null>(
    null,
  );
  const [defaultPair, setDefaultPair] = useState<[number, number] | null>(null);
  const [precomputedCorners, setPrecomputedCorners] = useState<Corners | null>(
    null,
  );
  const [jsonWidth, setJsonWidth] = useState(0);
  const [jsonHeight, setJsonHeight] = useState(0);
  const [imageSrc, setImageSrc] = useState('');
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [panels, setPanels] = useState<PanelPolygon[]>([]);
  const [panelLabels, setPanelLabels] = useState<string[] | undefined>(
    undefined,
  );
  const [boxes, setBoxes] = useState<Box[]>([]);
  // Detection rotations currently shown in boxes mode (0/90/270°).
  const [enabledAngles, setEnabledAngles] = useState<Set<number>>(new Set());
  const [adjacencyData, setAdjacencyData] = useState<AdjacencyData | null>(
    null,
  );
  // Stem of the loaded image's filename (e.g. "p49"), which identifies the page
  // in volume-level files like adjacency.json.
  const [imageStem, setImageStem] = useState<string | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );
  // The (volume, page) this view is on, when opened via a `?files=data/...`
  // deep link — the anchor for per-page notes. Null for dropped files.
  const [noteContext, setNoteContext] = useState<NoteContext | null>(null);

  // Display toggles.
  const [opacity, setOpacity] = useState(85); // 0..100
  const [showStreetsOnImage, setShowStreetsOnImage] = useState(true);
  const [showIntersectionsOnImage, setShowIntersectionsOnImage] =
    useState(true);
  const [colorByInlier, setColorByInlier] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showIntersections, setShowIntersections] = useState(true);
  const [showTruth, setShowTruth] = useState(false);
  const [filters, setFilters] = useState<DetectionFilters>({
    minConfidence: 0.15,
    minShortSide: 20,
    minLongSide: 20,
    showIgnored: false,
  });

  const prevObjectUrlRef = useRef<string | null>(null);

  // Interactive seed-pair exploration (only when the georef was written with --debug and
  // carries `gcp_pairs`). The active pair's fit is looked up — never recomputed — and its
  // precomputed corners drive label repositioning/recolouring below.
  const activePair = useMemo(
    () => findPairRecord(gcpPairs, selectedPair),
    [gcpPairs, selectedPair],
  );
  const overrideCorners =
    activePair && !activePair.degenerate ? (activePair.corners ?? null) : null;

  // Streets/intersections shown on the map: for a non-default pair, reposition labels through
  // the pair's corners and recolour by its inlier sets (pure rendering — no fit/scoring here).
  const displayStreets = useMemo<Street[]>(() => {
    if (!overrideCorners || !activePair) return streets;
    const inliers = new Set(activePair.inlier_streets ?? []);
    return streets.map((s, i) => {
      const [lon, lat] = projectThroughCorners(
        overrideCorners,
        jsonWidth,
        jsonHeight,
        s.x,
        s.y,
      );
      const [dLon, dLat] =
        s.dir_x !== undefined && s.dir_y !== undefined
          ? directionThroughCorners(
              overrideCorners,
              jsonWidth,
              jsonHeight,
              s.dir_x,
              s.dir_y,
            )
          : [s.dir_lon, s.dir_lat];
      return {
        ...s,
        lon,
        lat,
        dir_lon: dLon,
        dir_lat: dLat,
        inlier: inliers.has(i),
      };
    });
  }, [overrideCorners, activePair, streets, jsonWidth, jsonHeight]);

  const displayIntersections = useMemo<IntersectionPoint[]>(() => {
    if (!activePair) return intersections;
    const inliers = new Set(activePair.inlier_intersections ?? []);
    return intersections.map((ix, i) => ({
      ...ix,
      inlier: inliers.has(i),
      initial: i === activePair.a || i === activePair.b,
    }));
  }, [activePair, intersections]);

  const corners = useMemo(
    () =>
      overrideCorners ??
      precomputedCorners ??
      computeCorners(displayStreets, jsonWidth, jsonHeight),
    [
      overrideCorners,
      precomputedCorners,
      displayStreets,
      jsonWidth,
      jsonHeight,
    ],
  );

  // Precomputed stats for the GcpControls readout of the active pair.
  const gcpStats = useMemo<GcpFitStats | null>(() => {
    if (!selectedPair) return null;
    if (!activePair) return null;
    if (activePair.degenerate) {
      return {
        degenerate: true,
        numInliers: 0,
        numOutliers: 0,
        score: -Infinity,
        meanErrorM: null,
        maxErrorM: null,
      };
    }
    const numInliers = activePair.inlier_streets?.length ?? 0;
    return {
      degenerate: false,
      numInliers,
      numOutliers: streets.length - numInliers,
      score: activePair.score ?? 0,
      meanErrorM: activePair.mean_error_m ?? null,
      maxErrorM: activePair.max_error_m ?? null,
    };
  }, [selectedPair, activePair, streets.length]);

  const filteredDetections = useMemo(
    () => filterDetections(detections, filters),
    [detections, filters],
  );

  // Adjacency mode: the loaded image's page entry (identified by filename stem)
  // and its digit reads converted to the Detection shape the overlay renders.
  const adjacencyPage =
    mode === 'adjacency' && adjacencyData && imageStem
      ? (adjacencyData.pages[imageStem] ?? null)
      : null;
  const adjacencyDetections = useMemo<IndexedDetection[]>(() => {
    if (!adjacencyPage || !adjacencyData || !imageStem) return [];
    // Page numbers of this page's reciprocated neighbors; claims of these render blue.
    const mutualNumbers = new Set<number>();
    for (const [a, b] of adjacencyData.adjacency) {
      const other = a === imageStem ? b : b === imageStem ? a : null;
      const number = other ? adjacencyData.pages[other]?.number : null;
      if (number != null) mutualNumbers.add(number);
    }
    return adjacencyPage.detections.map((d, i) => ({
      det: detectionFromAdjacency(d, mutualNumbers),
      i,
    }));
  }, [adjacencyPage, adjacencyData, imageStem]);
  // Polygon coordinates live in the scanned image's pixel space; prefer its
  // recorded dimensions so a different-resolution image still lines up.
  const overlayWidth = adjacencyPage?.width ?? jsonWidth;
  const overlayHeight = adjacencyPage?.height ?? jsonHeight;

  // Parse georef JSON text and update streets/intersections/corners/dimensions.
  function applyGeorefJson(text: string): void {
    let data: GeorefData;
    try {
      data = JSON.parse(text) as GeorefData;
    } catch {
      return; // not valid JSON, skip update
    }
    const newIntersections = (data.intersections ?? []).map((ix) => ({
      ...ix,
    }));
    setStreets((data.streets ?? []).map((s) => ({ ...s })));
    setIntersections(newIntersections);
    setKeymap(data.keymap ?? null);
    setTruth(data.truth ?? null);
    setGcpPairs(data.gcp_pairs ?? null);
    // The pipeline's chosen pair drives the reset button; it is absent in a nofit debug file,
    // where we instead open on the best-scoring (still-rejected) pair so the explorer renders.
    const pipelinePair = data.gcp_pairs
      ? initialPairFrom(newIntersections)
      : null;
    setDefaultPair(pipelinePair);
    setSelectedPair(pipelinePair ?? bestPairFrom(data.gcp_pairs ?? null));
    setPrecomputedCorners(data.corners ?? null);
    if (data.width && data.height) {
      setJsonWidth(data.width);
      setJsonHeight(data.height);
    }
  }

  // Classify dropped JSON text and switch modes / load data accordingly.
  function processJson(
    text: string,
    fallbackWidth: number,
    fallbackHeight: number,
  ): void {
    const result = parseDroppedJson(text, {
      width: fallbackWidth,
      height: fallbackHeight,
    });
    if (result.kind === 'invalid') return;
    if (result.kind === 'streets') {
      setMode('streets');
      setSelectedIndices(new Set());
      setDetections(result.detections);
      setPanels([]);
      setPanelLabels(undefined);
      setBoxes([]);
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else if (result.kind === 'panels') {
      setMode('panels');
      setSelectedIndices(new Set());
      setPanels(result.panels);
      setPanelLabels(result.labels);
      setDetections([]);
      setBoxes([]);
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else if (result.kind === 'boxes') {
      setMode('boxes');
      setSelectedIndices(new Set());
      setBoxes(result.boxes);
      setEnabledAngles(new Set(result.boxes.map((box) => box.angle)));
      setDetections([]);
      setPanels([]);
      setPanelLabels(undefined);
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else if (result.kind === 'adjacency') {
      setMode('adjacency');
      setSelectedIndices(new Set());
      setAdjacencyData(result.data);
      setDetections([]);
      setPanels([]);
      setPanelLabels(undefined);
      setBoxes([]);
    } else {
      setMode('georef');
      setSelectedIndices(new Set());
      setDetections([]);
      setPanels([]);
      setPanelLabels(undefined);
      setBoxes([]);
      applyGeorefJson(result.text);
    }
  }

  // Toggle whether a detection rotation's boxes are shown (boxes mode).
  function toggleAngle(angle: number): void {
    setEnabledAngles((prev) => {
      const next = new Set(prev);
      if (next.has(angle)) next.delete(angle);
      else next.add(angle);
      return next;
    });
  }

  // Point the viewer at a decoded image, updating its source and JSON dimensions.
  function applyImage(el: HTMLImageElement, src: string): void {
    setImageEl(el);
    setImageSrc(src);
    setJsonWidth(el.naturalWidth);
    setJsonHeight(el.naturalHeight);
  }

  // Handle files dropped onto the image column (image and/or JSON).
  async function handleFiles(files: File[]): Promise<void> {
    const imageFile = files.find((f) => f.type.startsWith('image/'));
    const jsonFile = files.find((f) => f.name.endsWith('.json'));

    let fallbackWidth = jsonWidth;
    let fallbackHeight = jsonHeight;

    if (imageFile) {
      if (prevObjectUrlRef.current) {
        URL.revokeObjectURL(prevObjectUrlRef.current);
      }
      const url = URL.createObjectURL(imageFile);
      prevObjectUrlRef.current = url;
      const el = await loadImage(url);
      applyImage(el, url);
      setImageStem(pageStem(imageFile.name));
      fallbackWidth = el.naturalWidth;
      fallbackHeight = el.naturalHeight;
    }

    if (jsonFile) {
      const text = await jsonFile.text();
      processJson(text, fallbackWidth, fallbackHeight);
    }
  }

  // Load image and/or JSON data from dev-server URLs (the `?files=` deep link).
  // Mirrors handleFiles, but fetches served files instead of reading File blobs.
  async function loadFromUrls(files: string[]): Promise<void> {
    const imageFile = files.find(isImageUrl);
    const jsonFile = files.find((f) => f.endsWith('.json'));

    let fallbackWidth = jsonWidth;
    let fallbackHeight = jsonHeight;

    try {
      if (imageFile) {
        if (prevObjectUrlRef.current) {
          URL.revokeObjectURL(prevObjectUrlRef.current);
          prevObjectUrlRef.current = null;
        }
        const src = resolveDataUrl(imageFile);
        const el = await loadImage(src);
        applyImage(el, src);
        setImageStem(pageStem(imageFile));
        fallbackWidth = el.naturalWidth;
        fallbackHeight = el.naturalHeight;
      }

      if (jsonFile) {
        const response = await fetch(resolveDataUrl(jsonFile));
        if (!response.ok) {
          throw new Error(`${jsonFile}: HTTP ${response.status}`);
        }
        processJson(await response.text(), fallbackWidth, fallbackHeight);
      }
    } catch (err) {
      console.error('Failed to load files from URL:', err);
    }
  }

  // Expose a debug API on `window.mapsnap` for injecting data without the UI.
  // Re-registered each render so it always closes over the latest state.
  useEffect(() => {
    window.mapsnap = {
      loadJson: (text: string) => processJson(text, jsonWidth, jsonHeight),
      setImage: async (url: string) => applyImage(await loadImage(url), url),
    };
  });

  // On first load, honor a `?files=data/image.jpg,data/streets.json` deep link
  // by fetching those files from the dev server and entering the matching view.
  useEffect(() => {
    const filesParam = new URLSearchParams(window.location.search).get('files');
    if (!filesParam) return;
    const files = filesParam
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    setNoteContext(noteContextFromFiles(files));
    void loadFromUrls(files);
    // Runs once on mount; loadFromUrls closes over the initial (empty) state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cycle warped-image opacity through 0/50/100% on the 'p' key (georef mode),
  // unless the user is typing (e.g. in the note editor).
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'p' || mode !== 'georef' || isTypingTarget(e.target))
        return;
      const steps = [0, 50, 100];
      setOpacity((prev) => {
        const nextIdx = (steps.indexOf(prev) + 1) % steps.length;
        return steps[nextIdx] ?? steps[0];
      });
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, [mode]);

  if (mode === 'iiif') {
    return (
      <div className="container container-iiif">
        <VolumeViewer />
      </div>
    );
  }

  return (
    <div
      className={mode === 'panels' ? 'container container-panels' : 'container'}
    >
      <nav className="view-nav">
        <a href="?view=iiif">volume viewer</a>
        {noteContext && <NoteButton ctx={noteContext} />}
      </nav>
      <ImageColumn
        mode={mode}
        imageSrc={imageSrc}
        jsonWidth={overlayWidth}
        jsonHeight={overlayHeight}
        streets={streets}
        intersections={intersections}
        filteredDetections={
          mode === 'adjacency' ? adjacencyDetections : filteredDetections
        }
        panels={panels}
        panelLabels={panelLabels}
        boxes={boxes}
        enabledAngles={enabledAngles}
        selectedIndices={selectedIndices}
        onSelectIndices={setSelectedIndices}
        showStreetsOnImage={showStreetsOnImage}
        setShowStreetsOnImage={setShowStreetsOnImage}
        showIntersectionsOnImage={showIntersectionsOnImage}
        setShowIntersectionsOnImage={setShowIntersectionsOnImage}
        colorByInlier={colorByInlier}
        setColorByInlier={setColorByInlier}
        filters={filters}
        setFilters={setFilters}
        onFiles={handleFiles}
      />
      <div className="map-column">
        {mode === 'georef' && (
          <>
            <MapView
              streets={displayStreets}
              intersections={displayIntersections}
              corners={corners}
              keymap={keymap}
              truth={showTruth ? truth : null}
              imageSrc={imageSrc}
              opacity={opacity / 100}
              showLabels={showLabels}
              showIntersections={showIntersections}
              colorByInlier={colorByInlier}
            />
            {gcpPairs && selectedPair && intersections.length >= 2 && (
              <GcpControls
                intersections={intersections}
                selectedPair={selectedPair}
                onChange={setSelectedPair}
                defaultPair={defaultPair}
                result={gcpStats}
              />
            )}
            <div className="opacity-control">
              <label htmlFor="opacity-slider">Opacity</label>
              <input
                type="range"
                id="opacity-slider"
                min={0}
                max={100}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <span>{opacity}%</span>
            </div>
            <div className="opacity-control">
              <input
                type="checkbox"
                id="show-labels"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
              />
              <label htmlFor="show-labels">Show street labels on map</label>
            </div>
            <div className="opacity-control">
              <input
                type="checkbox"
                id="show-intersections"
                checked={showIntersections}
                onChange={(e) => setShowIntersections(e.target.checked)}
              />
              <label htmlFor="show-intersections">
                Show intersection GCPs on map
              </label>
            </div>
            <div className="opacity-control">
              <input
                type="checkbox"
                id="show-truth"
                checked={showTruth}
                disabled={!truth}
                onChange={(e) => setShowTruth(e.target.checked)}
              />
              <label htmlFor="show-truth">
                Show truth data on map
                {!truth && ' (none available)'}
              </label>
            </div>
          </>
        )}
        {mode === 'streets' && (
          <DetectionsTable
            detections={filteredDetections}
            selectedIndices={selectedIndices}
            onSelect={(index) => setSelectedIndices(new Set([index]))}
            image={imageEl}
            jsonWidth={jsonWidth}
            jsonHeight={jsonHeight}
          />
        )}
        {mode === 'panels' && (
          <PanelsTable
            panels={panels}
            panelLabels={panelLabels}
            selectedIndices={selectedIndices}
            onSelect={(index) => setSelectedIndices(new Set([index]))}
            jsonWidth={jsonWidth}
            jsonHeight={jsonHeight}
          />
        )}
        {mode === 'boxes' && (
          <BoxControls
            boxes={boxes}
            enabledAngles={enabledAngles}
            onToggleAngle={toggleAngle}
          />
        )}
        {mode === 'adjacency' && adjacencyData && (
          <AdjacencyTable
            adjacency={adjacencyData}
            imageStem={imageStem}
            detections={adjacencyDetections}
            selectedIndices={selectedIndices}
            onSelect={(index) => setSelectedIndices(new Set([index]))}
            image={imageEl}
            jsonWidth={overlayWidth}
            jsonHeight={overlayHeight}
          />
        )}
      </div>
    </div>
  );
}
