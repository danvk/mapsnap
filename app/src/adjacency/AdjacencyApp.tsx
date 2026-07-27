import { useEffect, useRef, useState } from 'react';
import '../keymap/keymap.css';
import { pointInPolygon } from '../geometry';
import { isTypingTarget } from '../keyboard';
import { useElementSize } from '../hooks/useElementSize';
import { loadImage } from '../loadImage';
import { BoxSizeControls } from '../keymap/BoxSizeControls';
import { ImageList } from '../keymap/ImageList';
import { LabelsOverlay } from '../keymap/LabelsOverlay';
import { LabelsTable } from '../keymap/LabelsTable';
import { createLabelsJson, labelBox } from '../keymap/labels';
import type { ImageInfo, Label } from '../keymap/types';
import {
  fetchLabels,
  fetchPages,
  fetchVolumes,
  pageImageUrl,
  saveLabels,
} from './api';

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

/**
 * Default label-box size for adjacency labels, in image pixels. The volume-root
 * pages are 25%-scale scans and the printed neighbor numbers along sheet edges
 * are small, so the box starts at half the key-map default; the sliders adjust
 * it live.
 */
const ADJACENCY_BOX_WIDTH = 80;
const ADJACENCY_BOX_HEIGHT = 54;

/**
 * Adjacency truth labeler: pick a volume, pick a page, click the centers of
 * the printed neighbor-page numbers along its edges, and type each number's
 * text. Labels save automatically to the volume's adjacency-truth.json.
 */
export function AdjacencyApp() {
  const [volumes, setVolumes] = useState<
    { name: string; pageCount: number; labeledPages: number }[]
  >([]);
  const [volume, setVolume] = useState<string | null>(null);
  const [pages, setPages] = useState<ImageInfo[]>([]);
  const [selectedPage, setSelectedPage] = useState<string | null>(null);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [labels, setLabels] = useState<Label[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');
  const [showOnlyUnlabeled, setShowOnlyUnlabeled] = useState(false);
  const [boxWidth, setBoxWidth] = useState(ADJACENCY_BOX_WIDTH);
  const [boxHeight, setBoxHeight] = useState(ADJACENCY_BOX_HEIGHT);

  const [imgRef, imgSize] = useElementSize<HTMLImageElement>();
  // Only persist labels that changed via user edits, not freshly loaded ones.
  const dirtyRef = useRef(false);
  // Mirror of `labels` so click handlers always see the latest list.
  const labelsRef = useRef<Label[]>([]);
  // Set by the "Next page" button so the incoming page's first text box takes
  // focus once its rows render — finishing a page is then tab+enter+type.
  // Deliberately NOT set by the n/p/j/k shortcuts: moving focus into a text
  // box would swallow the next keypress as typed text.
  const focusFirstOnLoadRef = useRef(false);

  // Load the volume list once.
  useEffect(() => {
    fetchVolumes().then(setVolumes).catch(console.error);
  }, []);

  // Load the selected volume's page list.
  useEffect(() => {
    if (!volume) return;
    let cancelled = false;
    setPages([]);
    setSelectedPage(null);
    fetchPages(volume)
      .then((next) => {
        if (!cancelled) setPages(next);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [volume]);

  // Load the selected page's image and existing labels.
  useEffect(() => {
    if (!volume || !selectedPage) return;
    let cancelled = false;
    dirtyRef.current = false;
    setSelectedIndex(null);
    setSaveStatus('idle');
    Promise.all([
      loadImage(pageImageUrl(volume, selectedPage)),
      fetchLabels(volume, selectedPage),
    ])
      .then(([el, entry]) => {
        if (cancelled) return;
        setImageEl(el);
        setImageWidth(entry?.width ?? el.naturalWidth);
        setImageHeight(entry?.height ?? el.naturalHeight);
        dirtyRef.current = false;
        labelsRef.current = entry?.labels ?? [];
        setLabels(labelsRef.current);
        if (focusFirstOnLoadRef.current) {
          focusFirstOnLoadRef.current = false;
          // After React commits the new rows; the top row is the newest label.
          requestAnimationFrame(() => {
            document
              .querySelector<HTMLInputElement>(
                '#detections-table input.label-text-input',
              )
              ?.focus();
          });
        }
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [volume, selectedPage]);

  // Persist edits (debounced). Skipped for freshly loaded data.
  useEffect(() => {
    if (!dirtyRef.current || !volume || !selectedPage) return;
    const handle = setTimeout(async () => {
      setSaveStatus('saving');
      try {
        await saveLabels(
          volume,
          selectedPage,
          createLabelsJson(imageWidth, imageHeight, labels),
        );
        setSaveStatus('saved');
        const withText = labels.filter((l) => l.text.trim()).length;
        setPages((prev) =>
          prev.map((info) =>
            info.name === selectedPage
              ? { ...info, withText, withoutText: labels.length - withText }
              : info,
          ),
        );
      } catch (err) {
        console.error(err);
        setSaveStatus('error');
      }
    }, 500);
    return () => clearTimeout(handle);
  }, [labels, volume, selectedPage, imageWidth, imageHeight]);

  // Step to an adjacent page in the list, clamped at the ends.
  function stepPage(delta: number): void {
    if (!pages.length) return;
    const index = pages.findIndex((info) => info.name === selectedPage);
    const next =
      index === -1
        ? delta > 0
          ? 0
          : pages.length - 1
        : Math.min(pages.length - 1, Math.max(0, index + delta));
    setSelectedPage(pages[next]!.name);
  }
  const pageIndex = pages.findIndex((info) => info.name === selectedPage);
  const hasNextPage = pageIndex !== -1 && pageIndex < pages.length - 1;

  // n/j and p/k step through the page list while no text field is focused.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey)
        return;
      const delta =
        e.key === 'n' || e.key === 'j'
          ? 1
          : e.key === 'p' || e.key === 'k'
            ? -1
            : 0;
      if (!delta) return;
      e.preventDefault();
      stepPage(delta);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  function editLabels(next: Label[]): void {
    dirtyRef.current = true;
    labelsRef.current = next;
    setLabels(next);
  }

  // Add a label at the click point, or select an existing one if clicked.
  function handleImageClick(e: React.MouseEvent): void {
    if (!selectedPage || !imgSize.width || !imgSize.height) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * imageWidth) / imgSize.width;
    const y = ((e.clientY - rect.top) * imageHeight) / imgSize.height;
    const current = labelsRef.current;
    const hitIndex = current.findIndex((label) =>
      pointInPolygon(x, y, labelBox(label.x, label.y, boxWidth, boxHeight)),
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
      <div className="image-list-column">
        <select
          className="volume-select"
          value={volume ?? ''}
          onChange={(e) => setVolume(e.target.value || null)}
        >
          <option value="">Select a volume…</option>
          {volumes.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name} ({v.labeledPages}/{v.pageCount})
            </option>
          ))}
        </select>
        <ImageList
          heading="Pages"
          images={pages}
          selectedName={selectedPage}
          onSelect={setSelectedPage}
        />
      </div>

      <div className="keymap-center">
        {volume && selectedPage ? (
          <div
            className="image-wrapper"
            style={{ cursor: 'crosshair' }}
            onClick={handleImageClick}
          >
            <img
              ref={imgRef}
              src={pageImageUrl(volume, selectedPage)}
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
              boxWidth={boxWidth}
              boxHeight={boxHeight}
              displayWidth={imgSize.width}
              displayHeight={imgSize.height}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
            />
          </div>
        ) : (
          <div className="keymap-empty">
            {volume
              ? 'Select a page to label its printed neighbor numbers'
              : 'Select a volume to begin'}
          </div>
        )}
      </div>

      <div className="keymap-right">
        <div className="keymap-status">
          {selectedPage && (
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
        <BoxSizeControls
          boxWidth={boxWidth}
          boxHeight={boxHeight}
          onChangeWidth={setBoxWidth}
          onChangeHeight={setBoxHeight}
        />
        <LabelsTable
          labels={labels}
          selectedIndex={selectedIndex}
          showOnlyUnlabeled={showOnlyUnlabeled}
          image={imageEl}
          boxWidth={boxWidth}
          boxHeight={boxHeight}
          onSelect={setSelectedIndex}
          onChangeText={handleChangeText}
          onDelete={handleDelete}
          onNextPage={
            hasNextPage
              ? () => {
                  focusFirstOnLoadRef.current = true;
                  stepPage(1);
                }
              : undefined
          }
        />
      </div>
    </div>
  );
}
