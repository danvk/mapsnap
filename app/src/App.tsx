import { useEffect, useMemo, useRef, useState } from 'react';
import './styles.css';
import type {
  Corners,
  GeorefData,
  IntersectionPoint,
  PanelPolygon,
  Street,
  Detection,
} from './types';
import { computeCorners } from './geometry';
import { filterDetections, type DetectionFilters } from './detections';
import { parseDroppedJson } from './fileLoading';
import { ImageColumn, type Mode } from './components/ImageColumn';
import { MapView } from './components/MapView';
import { DetectionsTable } from './components/DetectionsTable';
import { PanelsTable } from './components/PanelsTable';
import { loadImage } from './loadImage';

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
 * Top-level debugger app with three modes: georef (map + overlays), streets
 * (text detections), and panels (page-split polygons).
 */
export function App() {
  const [mode, setMode] = useState<Mode>('georef');
  const [streets, setStreets] = useState<Street[]>([]);
  const [intersections, setIntersections] = useState<IntersectionPoint[]>([]);
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
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );

  // Display toggles.
  const [opacity, setOpacity] = useState(85); // 0..100
  const [showStreetsOnImage, setShowStreetsOnImage] = useState(true);
  const [showIntersectionsOnImage, setShowIntersectionsOnImage] =
    useState(true);
  const [colorByInlier, setColorByInlier] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showIntersections, setShowIntersections] = useState(true);
  const [filters, setFilters] = useState<DetectionFilters>({
    minConfidence: 0.15,
    minShortSide: 20,
    minLongSide: 20,
    showIgnored: false,
  });

  const prevObjectUrlRef = useRef<string | null>(null);

  const corners = useMemo(
    () => precomputedCorners ?? computeCorners(streets, jsonWidth, jsonHeight),
    [precomputedCorners, streets, jsonWidth, jsonHeight],
  );
  const filteredDetections = useMemo(
    () => filterDetections(detections, filters),
    [detections, filters],
  );

  // Parse georef JSON text and update streets/intersections/corners/dimensions.
  function applyGeorefJson(text: string): void {
    let data: GeorefData;
    try {
      data = JSON.parse(text) as GeorefData;
    } catch {
      return; // not valid JSON, skip update
    }
    setStreets((data.streets ?? []).map((s) => ({ ...s })));
    setIntersections((data.intersections ?? []).map((ix) => ({ ...ix })));
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
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else if (result.kind === 'panels') {
      setMode('panels');
      setSelectedIndices(new Set());
      setPanels(result.panels);
      setPanelLabels(result.labels);
      setDetections([]);
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else {
      setMode('georef');
      setSelectedIndices(new Set());
      setDetections([]);
      setPanels([]);
      setPanelLabels(undefined);
      applyGeorefJson(result.text);
    }
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
      setImageEl(el);
      setImageSrc(url);
      fallbackWidth = el.naturalWidth;
      fallbackHeight = el.naturalHeight;
      setJsonWidth(el.naturalWidth);
      setJsonHeight(el.naturalHeight);
    }

    if (jsonFile) {
      const text = await jsonFile.text();
      processJson(text, fallbackWidth, fallbackHeight);
    }
  }

  // Expose a debug API on `window.mapsnap` for injecting data without the UI.
  // Re-registered each render so it always closes over the latest state.
  useEffect(() => {
    window.mapsnap = {
      loadJson: (text: string) => processJson(text, jsonWidth, jsonHeight),
      setImage: async (url: string) => {
        const el = await loadImage(url);
        setImageEl(el);
        setImageSrc(url);
        setJsonWidth(el.naturalWidth);
        setJsonHeight(el.naturalHeight);
      },
    };
  });

  // Cycle warped-image opacity through 0/50/100% on the 'p' key (georef mode).
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'p' || mode !== 'georef') return;
      const steps = [0, 50, 100];
      setOpacity((prev) => {
        const nextIdx = (steps.indexOf(prev) + 1) % steps.length;
        return steps[nextIdx] ?? steps[0];
      });
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, [mode]);

  return (
    <div
      className={mode === 'panels' ? 'container container-panels' : 'container'}
    >
      <ImageColumn
        mode={mode}
        imageSrc={imageSrc}
        jsonWidth={jsonWidth}
        jsonHeight={jsonHeight}
        streets={streets}
        intersections={intersections}
        filteredDetections={filteredDetections}
        panels={panels}
        panelLabels={panelLabels}
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
              streets={streets}
              intersections={intersections}
              corners={corners}
              imageSrc={imageSrc}
              opacity={opacity / 100}
              showLabels={showLabels}
              showIntersections={showIntersections}
              colorByInlier={colorByInlier}
            />
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
      </div>
    </div>
  );
}
