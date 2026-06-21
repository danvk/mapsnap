import { useEffect, useRef, useState } from 'react';
import './keymap.css';
import { pointInPolygon } from '../geometry';
import { useElementSize } from '../hooks/useElementSize';
import { loadImage } from '../loadImage';
import { fetchImages, fetchLabels, imageUrl, saveLabels } from './api';
import { createLabelsJson, labelBox, labelBoxSize } from './labels';
import { ImageList } from './ImageList';
import { LabelsOverlay } from './LabelsOverlay';
import { LabelsTable } from './LabelsTable';
import type { ImageInfo, Label } from './types';

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

/**
 * Key map truth-data labeler: pick an image, click to drop labels, type the
 * text for each, and have the labels.json sidecar saved automatically.
 */
export function KeymapApp() {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [labels, setLabels] = useState<Label[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');
  const [showOnlyUnlabeled, setShowOnlyUnlabeled] = useState(false);

  const [imgRef, imgSize] = useElementSize<HTMLImageElement>();
  // Only persist labels that changed via user edits, not freshly loaded ones.
  const dirtyRef = useRef(false);
  // Mirror of `labels` so click handlers always see the latest list, even when
  // several edits happen before React re-renders.
  const labelsRef = useRef<Label[]>([]);

  // Load the list of available images once.
  useEffect(() => {
    fetchImages().then(setImages).catch(console.error);
  }, []);

  // Load the selected image and its existing labels.
  useEffect(() => {
    if (!selectedName) return;
    let cancelled = false;
    dirtyRef.current = false;
    setSelectedIndex(null);
    setSaveStatus('idle');
    Promise.all([loadImage(imageUrl(selectedName)), fetchLabels(selectedName)])
      .then(([el, sidecar]) => {
        if (cancelled) return;
        setImageEl(el);
        setImageWidth(sidecar?.width ?? el.naturalWidth);
        setImageHeight(sidecar?.height ?? el.naturalHeight);
        dirtyRef.current = false;
        labelsRef.current = sidecar?.labels ?? [];
        setLabels(labelsRef.current);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [selectedName]);

  // Persist edits to the sidecar (debounced). Skipped for freshly loaded data.
  useEffect(() => {
    if (!dirtyRef.current || !selectedName) return;
    const handle = setTimeout(async () => {
      setSaveStatus('saving');
      try {
        await saveLabels(
          selectedName,
          createLabelsJson(selectedName, imageWidth, imageHeight, labels),
        );
        setSaveStatus('saved');
        setImages((prev) =>
          prev.map((info) =>
            info.name === selectedName
              ? { ...info, labelCount: labels.length }
              : info,
          ),
        );
      } catch (err) {
        console.error(err);
        setSaveStatus('error');
      }
    }, 500);
    return () => clearTimeout(handle);
  }, [labels, selectedName, imageWidth, imageHeight]);

  // Apply a user edit to the labels and mark them dirty so they get saved.
  function editLabels(next: Label[]): void {
    dirtyRef.current = true;
    labelsRef.current = next;
    setLabels(next);
  }

  // Box size scaled to the image's resolution (full vs. 25%-scale).
  const box = labelBoxSize(imageWidth, imageHeight);

  // Add a label at the click point, or select an existing one if clicked.
  function handleImageClick(e: React.MouseEvent): void {
    if (!selectedName || !imgSize.width || !imgSize.height) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * imageWidth) / imgSize.width;
    const y = ((e.clientY - rect.top) * imageHeight) / imgSize.height;
    const current = labelsRef.current;
    const hitIndex = current.findIndex((label) =>
      pointInPolygon(x, y, labelBox(label.x, label.y, box.width, box.height)),
    );
    if (hitIndex >= 0) {
      setSelectedIndex(hitIndex);
      return;
    }
    editLabels([...current, { x, y, text: '' }]);
    setSelectedIndex(current.length);
  }

  function handleChangeText(index: number, text: string): void {
    editLabels(
      labelsRef.current.map((label, i) =>
        i === index ? { ...label, text } : label,
      ),
    );
  }

  function handleDelete(index: number): void {
    editLabels(labelsRef.current.filter((_, i) => i !== index));
    setSelectedIndex((prev) => {
      if (prev === null) return null;
      if (prev === index) return null;
      return prev > index ? prev - 1 : prev;
    });
  }

  const statusText: Record<SaveStatus, string> = {
    idle: '',
    saving: 'Saving…',
    saved: 'Saved',
    error: 'Save failed',
  };

  return (
    <div className="keymap-container">
      <ImageList
        images={images}
        selectedName={selectedName}
        onSelect={setSelectedName}
      />

      <div className="keymap-center">
        {selectedName ? (
          <div
            className="image-wrapper"
            style={{ cursor: 'crosshair' }}
            onClick={handleImageClick}
          >
            <img
              ref={imgRef}
              src={imageUrl(selectedName)}
              className="keymap-image"
              style={
                imageWidth && imageHeight
                  ? { aspectRatio: `${imageWidth} / ${imageHeight}` }
                  : undefined
              }
            />
            <LabelsOverlay
              labels={labels}
              selectedIndex={selectedIndex}
              boxWidth={box.width}
              boxHeight={box.height}
              displayWidth={imgSize.width}
              displayHeight={imgSize.height}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
            />
          </div>
        ) : (
          <div className="keymap-empty">Select a key map to begin labeling</div>
        )}
      </div>

      <div className="keymap-right">
        <div className="keymap-status">
          {selectedName && (
            <>
              <span>{labels.length} labels</span>
              <span className={`save-status save-${saveStatus}`}>
                {statusText[saveStatus]}
              </span>
            </>
          )}
        </div>
        <label className="keymap-controls">
          <input
            type="checkbox"
            checked={showOnlyUnlabeled}
            onChange={(e) => setShowOnlyUnlabeled(e.target.checked)}
          />
          Only show labels without text
        </label>
        <LabelsTable
          labels={labels}
          selectedIndex={selectedIndex}
          showOnlyUnlabeled={showOnlyUnlabeled}
          image={imageEl}
          boxWidth={box.width}
          boxHeight={box.height}
          onSelect={setSelectedIndex}
          onChangeText={handleChangeText}
          onDelete={handleDelete}
        />
      </div>
    </div>
  );
}
