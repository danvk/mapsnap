import { useEffect, useMemo, useRef, useState } from 'react';
import './regions.css';
import { isTypingTarget } from '../keyboard';
import { loadImage } from '../loadImage';
import { ImageList } from '../keymap/ImageList';
import type { ImageInfo } from '../keymap/types';
import {
  fetchDetections,
  fetchImages,
  fetchRegions,
  imageUrl,
  saveRegions,
} from './api';
import { MIN_RING_VERTICES, autoLabel } from './polygons';
import { PolygonCanvas } from './PolygonCanvas';
import { RegionsTable } from './RegionsTable';
import type { KeymapDetection, RegionPolygon } from './types';

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 2;

/**
 * Key-map page-region labeler: pick a sheet, trace the coloured block around each
 * page number, and have a `<stem>.regions.panels.json` written under `raw/truth/`.
 *
 * The output is deliberately the same schema `page_regions` produces, so a traced
 * sheet can be scored against the detected segmentation — or against the footprints
 * projected from OIM fits — with `mapsnap.keymap.score_regions --truth`.
 *
 * Tracing 40 blocks would mean typing 40 page keys, so a closed ring takes its key
 * from whichever detected page number it encloses (see {@link autoLabel}); the table
 * is for the ones that stay ambiguous.
 */
export function RegionsApp() {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [regions, setRegions] = useState<RegionPolygon[]>([]);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [detections, setDetections] = useState<KeymapDetection[]>([]);
  const [showDetections, setShowDetections] = useState(true);
  const [zoom, setZoom] = useState(0.25);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');

  // Only persist regions that changed through user edits, not freshly loaded ones.
  const dirtyRef = useRef(false);
  // Mirrors of the live state, so keyboard handlers registered once always see it.
  const regionsRef = useRef<RegionPolygon[]>([]);
  const draftRef = useRef<[number, number][]>([]);
  const selectedRef = useRef<number | null>(null);
  regionsRef.current = regions;
  draftRef.current = draft;
  selectedRef.current = selectedIndex;

  useEffect(() => {
    fetchImages().then(setImages).catch(console.error);
  }, []);

  // Load the selected sheet, its existing regions, and its detected page numbers.
  useEffect(() => {
    if (!selectedName) return;
    let cancelled = false;
    dirtyRef.current = false;
    setSelectedIndex(null);
    setDraft([]);
    setSaveStatus('idle');
    Promise.all([
      loadImage(imageUrl(selectedName)),
      fetchRegions(selectedName),
      fetchDetections(selectedName).catch(() => []),
    ])
      .then(([element, sidecar, points]) => {
        if (cancelled) return;
        setImageWidth(sidecar?.width ?? element.naturalWidth);
        setImageHeight(sidecar?.height ?? element.naturalHeight);
        setRegions(sidecar?.regions ?? []);
        setDetections(points);
        dirtyRef.current = false;
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [selectedName]);

  // Persist edits (debounced), skipping freshly loaded data.
  useEffect(() => {
    if (!dirtyRef.current || !selectedName) return;
    const handle = setTimeout(() => {
      setSaveStatus('saving');
      saveRegions(selectedName, imageWidth, imageHeight, regions)
        .then(() => setSaveStatus('saved'))
        .catch((error) => {
          console.error(error);
          setSaveStatus('error');
        });
    }, 600);
    return () => clearTimeout(handle);
  }, [regions, selectedName, imageWidth, imageHeight]);

  function update(next: RegionPolygon[]): void {
    dirtyRef.current = true;
    setRegions(next);
  }

  function closeDraft(): void {
    const ring = draftRef.current;
    if (ring.length < MIN_RING_VERTICES) return;
    const taken = new Set(
      regionsRef.current.map((r) => r.text).filter((t) => t !== ''),
    );
    update([
      ...regionsRef.current,
      { ring, text: autoLabel(ring, detections, taken) },
    ]);
    setSelectedIndex(regionsRef.current.length);
    setDraft([]);
  }

  // One keyboard handler for the drawing verbs; inputs keep their own typing.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      if (isTypingTarget(event.target)) return;
      if (event.key === 'Enter') {
        closeDraft();
      } else if (event.key === 'Escape') {
        setDraft([]);
      } else if (event.key === 'Backspace' && draftRef.current.length > 0) {
        event.preventDefault();
        setDraft(draftRef.current.slice(0, -1));
      } else if (
        (event.key === 'Delete' || event.key === 'Backspace') &&
        draftRef.current.length === 0 &&
        selectedRef.current !== null
      ) {
        event.preventDefault();
        const index = selectedRef.current;
        update(regionsRef.current.filter((_, i) => i !== index));
        setSelectedIndex(null);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // closeDraft closes over `detections`, so re-register when those load.
  }, [detections]);

  const duplicates = useMemo(() => {
    const seen = new Map<string, number>();
    for (const region of regions) {
      if (region.text === '') continue;
      seen.set(region.text, (seen.get(region.text) ?? 0) + 1);
    }
    return new Set(
      [...seen.entries()].filter(([, n]) => n > 1).map(([text]) => text),
    );
  }, [regions]);

  const unlabeled = regions.filter((r) => r.text === '').length;

  return (
    <div className="regions-app">
      <div className="regions-sidebar">
        <ImageList
          images={images}
          selectedName={selectedName}
          onSelect={setSelectedName}
        />
      </div>

      <div className="regions-main">
        <div className="regions-toolbar">
          <span className="regions-title">
            {selectedName ?? 'pick a key map'}
          </span>
          <label>
            Zoom
            <input
              type="range"
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={0.01}
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
            />
            <span className="regions-zoom">{(zoom * 100).toFixed(0)}%</span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={showDetections}
              onChange={(e) => setShowDetections(e.target.checked)}
            />
            Show page numbers
          </label>
          <span className="regions-count">
            {regions.length} region{regions.length === 1 ? '' : 's'}
            {unlabeled > 0 && `, ${unlabeled} unnamed`}
            {duplicates.size > 0 && `, ${duplicates.size} duplicated`}
          </span>
          <span className={`regions-save regions-save-${saveStatus}`}>
            {saveStatus === 'saving'
              ? 'saving…'
              : saveStatus === 'saved'
                ? 'saved'
                : saveStatus === 'error'
                  ? 'save failed'
                  : ''}
          </span>
        </div>

        <p className="regions-help">
          Click to drop vertices; click the first vertex again, double-click, or
          press Enter to close. Escape abandons the ring, Backspace removes its
          last vertex. Click a region to select it, drag its handles to adjust,
          Delete to remove it.
        </p>

        {selectedName && imageWidth > 0 && (
          <PolygonCanvas
            imageSrc={imageUrl(selectedName)}
            width={imageWidth}
            height={imageHeight}
            zoom={zoom}
            regions={regions}
            draft={draft}
            selectedIndex={selectedIndex}
            detections={detections}
            showDetections={showDetections}
            onAddVertex={(point) => setDraft([...draftRef.current, point])}
            onCloseDraft={closeDraft}
            onSelect={setSelectedIndex}
            onMoveVertex={(region, vertex, to) => {
              const next = regionsRef.current.map((r, i) =>
                i === region
                  ? {
                      ...r,
                      ring: r.ring.map((p, j) => (j === vertex ? to : p)),
                    }
                  : r,
              );
              update(next);
            }}
          />
        )}
      </div>

      <RegionsTable
        regions={regions}
        width={imageWidth}
        height={imageHeight}
        selectedIndex={selectedIndex}
        duplicates={duplicates}
        onSelect={setSelectedIndex}
        onSetText={(index, text) =>
          update(
            regionsRef.current.map((r, i) =>
              i === index ? { ...r, text } : r,
            ),
          )
        }
        onDelete={(index) => {
          update(regionsRef.current.filter((_, i) => i !== index));
          setSelectedIndex(null);
        }}
      />
    </div>
  );
}
