import { useEffect, useMemo, useRef, useState } from 'react';
import './styles.css';
import type {
  Corners,
  GeorefData,
  IntersectionPoint,
  Street,
  Detection,
} from './types';
import { computeCorners } from './geometry';
import { filterDetections, type DetectionFilters } from './detections';
import { parseDroppedJson } from './fileLoading';
import { ImageColumn } from './components/ImageColumn';
import { MapView } from './components/MapView';
import { DetectionsTable } from './components/DetectionsTable';

type Mode = 'georef' | 'streets';

// Load an image element from a URL, resolving once it has decoded.
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = src;
  });
}

// Serialize georef state for display in the textarea.
function serializeGeoref(
  width: number,
  height: number,
  corners: Corners | null,
  streets: Street[],
  intersections: IntersectionPoint[],
): string {
  return JSON.stringify(
    { width, height, corners, streets, intersections },
    null,
    2,
  );
}

/** Top-level debugger app: georef mode (map + overlays) and streets mode (detections). */
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
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );
  const [jsonText, setJsonText] = useState(() =>
    serializeGeoref(0, 0, null, [], []),
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
      return; // invalid JSON mid-edit, skip update
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
      setDetections(result.detections);
      setSelectedIndices(new Set());
      setMode('streets');
      setJsonText(result.text);
      setJsonWidth(result.width);
      setJsonHeight(result.height);
    } else {
      setMode('georef');
      setDetections([]);
      setSelectedIndices(new Set());
      setJsonText(result.text);
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
      if (!jsonFile && mode === 'georef') {
        setJsonText(
          serializeGeoref(
            el.naturalWidth,
            el.naturalHeight,
            precomputedCorners,
            streets,
            intersections,
          ),
        );
      }
    }

    if (jsonFile) {
      const text = await jsonFile.text();
      processJson(text, fallbackWidth, fallbackHeight);
    }
  }

  // Handle JSON dropped onto the textarea (georef data replacement).
  async function handleTextareaDrop(e: React.DragEvent): Promise<void> {
    const file = [...e.dataTransfer.files].find((f) =>
      f.name.endsWith('.json'),
    );
    if (!file) return;
    e.preventDefault();
    const text = await file.text();
    processJson(text, jsonWidth, jsonHeight);
  }

  function handleTextareaChange(value: string): void {
    setJsonText(value);
    if (mode === 'streets') return;
    applyGeorefJson(value);
  }

  // Cycle warped-image opacity through 0/50/100% on the 'p' key (georef mode).
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'p' || mode === 'streets') return;
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
    <div className="container">
      <textarea
        rows={50}
        cols={40}
        value={jsonText}
        onChange={(e) => handleTextareaChange(e.target.value)}
        onDragOver={(e) => {
          if ([...e.dataTransfer.types].includes('Files')) e.preventDefault();
        }}
        onDrop={handleTextareaDrop}
      />
      <ImageColumn
        mode={mode}
        imageSrc={imageSrc}
        jsonWidth={jsonWidth}
        jsonHeight={jsonHeight}
        streets={streets}
        intersections={intersections}
        filteredDetections={filteredDetections}
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
        {mode === 'georef' ? (
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
        ) : (
          <DetectionsTable
            detections={filteredDetections}
            selectedIndices={selectedIndices}
            onSelect={(index) => setSelectedIndices(new Set([index]))}
            image={imageEl}
            jsonWidth={jsonWidth}
            jsonHeight={jsonHeight}
          />
        )}
      </div>
    </div>
  );
}
