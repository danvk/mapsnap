import { useState } from 'react';
import type { DetectionFilters, IndexedDetection } from '../detections';
import { pointInPolygon } from '../geometry';
import type { IntersectionPoint, PanelPolygon, Street } from '../types';
import { useElementSize } from '../hooks/useElementSize';
import { DetectionsOverlay } from './DetectionsOverlay';
import { GeorefOverlay } from './GeorefOverlay';
import { PanelsOverlay } from './PanelsOverlay';

export type Mode = 'georef' | 'streets' | 'panels' | 'adjacency';

interface ImageColumnProps {
  mode: Mode;
  imageSrc: string;
  jsonWidth: number;
  jsonHeight: number;
  streets: Street[];
  intersections: IntersectionPoint[];
  filteredDetections: IndexedDetection[];
  panels: PanelPolygon[];
  panelLabels?: string[];
  selectedIndices: Set<number>;
  onSelectIndices: (indices: Set<number>) => void;
  showStreetsOnImage: boolean;
  setShowStreetsOnImage: (value: boolean) => void;
  showIntersectionsOnImage: boolean;
  setShowIntersectionsOnImage: (value: boolean) => void;
  colorByInlier: boolean;
  setColorByInlier: (value: boolean) => void;
  filters: DetectionFilters;
  setFilters: (filters: DetectionFilters) => void;
  onFiles: (files: File[]) => void;
}

/**
 * Left column: the source image with its mode-specific SVG overlay (georef
 * streets/intersections, streets detections, or panel polygons), image-side
 * display toggles, and (in streets mode) the detection filter sliders. Accepts
 * dropped image and JSON files anywhere in the column.
 */
export function ImageColumn(props: ImageColumnProps) {
  const {
    mode,
    imageSrc,
    jsonWidth,
    jsonHeight,
    streets,
    intersections,
    filteredDetections,
    panels,
    panelLabels,
    selectedIndices,
    onSelectIndices,
    showStreetsOnImage,
    setShowStreetsOnImage,
    showIntersectionsOnImage,
    setShowIntersectionsOnImage,
    colorByInlier,
    setColorByInlier,
    filters,
    setFilters,
    onFiles,
  } = props;

  const [imgRef, imgSize] = useElementSize<HTMLImageElement>();
  const [dragOver, setDragOver] = useState(false);
  const streetsMode = mode === 'streets';
  const panelsMode = mode === 'panels';
  const georefMode = mode === 'georef';
  // Adjacency mode renders detections just like streets mode (overlay + click select).
  const detectionsMode = streetsMode || mode === 'adjacency';

  // In streets/panels mode, select the shapes under the click point. The wrapper
  // (currentTarget) tightly wraps the image, so its rect matches the image's.
  function handleClick(e: React.MouseEvent): void {
    if (georefMode || !imgSize.width || !imgSize.height) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) * jsonWidth) / imgSize.width;
    const imgY = ((e.clientY - rect.top) * jsonHeight) / imgSize.height;
    const hit = panelsMode
      ? panels.flatMap((polygon, i) =>
          pointInPolygon(imgX, imgY, polygon) ? [i] : [],
        )
      : filteredDetections
          .filter(({ det }) => pointInPolygon(imgX, imgY, det.polygon))
          .map(({ i }) => i);
    onSelectIndices(new Set(hit));
  }

  function handleDrop(e: React.DragEvent): void {
    setDragOver(false);
    const files = [...e.dataTransfer.files];
    const hasImage = files.some((f) => f.type.startsWith('image/'));
    const hasJson = files.some((f) => f.name.endsWith('.json'));
    if (!hasImage && !hasJson) return;
    e.preventDefault();
    onFiles(files);
  }

  function handleDragOver(e: React.DragEvent): void {
    if ([...e.dataTransfer.types].includes('Files')) {
      e.preventDefault();
      setDragOver(true);
    }
  }

  return (
    <div
      className="image-column"
      onDragOver={handleDragOver}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {imageSrc ? (
        <div
          className="image-wrapper"
          style={{ cursor: georefMode ? undefined : 'crosshair' }}
          onClick={handleClick}
        >
          <img
            ref={imgRef}
            src={imageSrc}
            style={{ aspectRatio: `${jsonWidth} / ${jsonHeight}` }}
          />
          {detectionsMode && (
            <DetectionsOverlay
              detections={filteredDetections}
              selectedIndices={selectedIndices}
              displayWidth={imgSize.width}
              displayHeight={imgSize.height}
              jsonWidth={jsonWidth}
              jsonHeight={jsonHeight}
            />
          )}
          {panelsMode && (
            <PanelsOverlay
              panels={panels}
              labels={panelLabels}
              selectedIndices={selectedIndices}
              displayWidth={imgSize.width}
              displayHeight={imgSize.height}
              jsonWidth={jsonWidth}
              jsonHeight={jsonHeight}
            />
          )}
          {georefMode && (
            <GeorefOverlay
              streets={streets}
              intersections={intersections}
              showStreetsOnImage={showStreetsOnImage}
              showIntersectionsOnImage={showIntersectionsOnImage}
              colorByInlier={colorByInlier}
              displayWidth={imgSize.width}
              displayHeight={imgSize.height}
              jsonWidth={jsonWidth}
              jsonHeight={jsonHeight}
            />
          )}
        </div>
      ) : (
        <div className={`drop-placeholder${dragOver ? ' drag-over' : ''}`}>
          Drop image here
        </div>
      )}

      {georefMode && (
        <>
          <div className="image-controls">
            <input
              type="checkbox"
              id="show-streets-on-image"
              checked={showStreetsOnImage}
              onChange={(e) => setShowStreetsOnImage(e.target.checked)}
            />
            <label htmlFor="show-streets-on-image">Show streets on image</label>
          </div>
          <div className="image-controls">
            <input
              type="checkbox"
              id="show-intersections-on-image"
              checked={showIntersectionsOnImage}
              onChange={(e) => setShowIntersectionsOnImage(e.target.checked)}
            />
            <label htmlFor="show-intersections-on-image">
              Show intersections on image
            </label>
          </div>
          <div className="image-controls">
            <input
              type="checkbox"
              id="color-by-inlier"
              checked={colorByInlier}
              onChange={(e) => setColorByInlier(e.target.checked)}
            />
            <label htmlFor="color-by-inlier">
              Color inliers/outliers differently
            </label>
          </div>
        </>
      )}

      {streetsMode && (
        <div id="detection-filters">
          <div className="filter-row">
            <label htmlFor="filter-confidence">
              Min confidence: <span>{filters.minConfidence.toFixed(3)}</span>
            </label>
            <input
              type="range"
              id="filter-confidence"
              min={0}
              max={1}
              step={0.001}
              value={filters.minConfidence}
              onChange={(e) =>
                setFilters({
                  ...filters,
                  minConfidence: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div className="filter-row">
            <label htmlFor="filter-short-side">
              Min short side: <span>{filters.minShortSide.toFixed(0)}</span>
            </label>
            <input
              type="range"
              id="filter-short-side"
              min={0}
              max={200}
              step={1}
              value={filters.minShortSide}
              onChange={(e) =>
                setFilters({
                  ...filters,
                  minShortSide: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div className="filter-row">
            <label htmlFor="filter-long-side">
              Min long side: <span>{filters.minLongSide.toFixed(0)}</span>
            </label>
            <input
              type="range"
              id="filter-long-side"
              min={0}
              max={200}
              step={1}
              value={filters.minLongSide}
              onChange={(e) =>
                setFilters({
                  ...filters,
                  minLongSide: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div className="filter-row">
            <input
              type="checkbox"
              id="show-ignored"
              checked={filters.showIgnored}
              onChange={(e) =>
                setFilters({ ...filters, showIgnored: e.target.checked })
              }
            />
            <label htmlFor="show-ignored">Show ignored detections</label>
          </div>
        </div>
      )}
    </div>
  );
}
