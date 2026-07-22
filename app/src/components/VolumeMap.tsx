import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';
import type { FeatureCollection } from 'geojson';
import { pointInPolygon } from '../geometry';
import type { PageGeo } from '../iiif/pages';
import type { AdjacencyClaim } from '../iiif/adjacency';

interface VolumeMapProps {
  /** Rewritten Georeference AnnotationPage to display, or null before load. */
  annotation: unknown;
  /** Derived page geometry for hit-testing and outlines. */
  pages: PageGeo[];
  /** Truth pages never fitted; drawn as dashed truth-location footprints. */
  missingPages: PageGeo[];
  /**
   * All truth pages (from the volume's main.iiif.json), for drawing the selected fit page's
   * truth box beneath its generated outline. Empty when the volume has no truth data.
   */
  truthPages: PageGeo[];
  /** Whether the missing-page footprints are drawn and clickable. */
  showMissing: boolean;
  /** itemIndex of the selected page, or null for no selection. */
  selectedItemIndex: number | null;
  /** Called with the clicked page's itemIndex, or null for empty space. */
  onSelectPage: (itemIndex: number | null) => void;
  /** Warped-image opacity in [0, 1]. */
  opacity: number;
  /**
   * True while a selected annotation is still loading. The map stays hidden
   * until it has been fit to that annotation, so it never flashes the default
   * location before the volume's real one arrives.
   */
  awaitingView: boolean;
  /** Per-itemIndex fill color for RMSE color-coding, or null when off. */
  pageColors: Map<number, string> | null;
  /** Adjacency claim boxes to draw (blue mutual / amber one-sided); empty to hide the overlay. */
  adjacencyClaims: AdjacencyClaim[];
  /**
   * File stem of the selected page, or null when nothing is selected. Claims on this page draw
   * filled; claims on other pages draw as a dashed outline, as if underneath it.
   */
  selectedStem: string | null;
  /** Called with per-page add results whenever a new annotation is shown. */
  onLoadResult?: (result: { loaded: number; failed: number }) => void;
}

const EMPTY_FEATURES: FeatureCollection = {
  type: 'FeatureCollection',
  features: [],
};

// The RMSE color-coding fill opacity at full slider opacity; scaled down in
// proportion as the opacity slider is reduced, so the fills fade with the maps.
const RMSE_FILL_OPACITY = 0.45;

// Overlay features for a selected page: its truth box(es) (green, drawn under everything), the
// full image rectangle (solid), the clipping polygon (dashed), and its GCPs (circles),
// distinguished by `kind`. `truthRings` are the matching truth pages' corner rectangles.
function selectionFeatures(
  page: PageGeo,
  truthRings: [number, number][][],
): FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [
      ...truthRings.map((ring): FeatureCollection['features'][0] => ({
        type: 'Feature',
        properties: { kind: 'truth' },
        geometry: { type: 'LineString', coordinates: ring },
      })),
      {
        type: 'Feature',
        properties: { kind: 'rect' },
        geometry: { type: 'LineString', coordinates: page.rectRing },
      },
      {
        type: 'Feature',
        properties: { kind: 'clip' },
        geometry: { type: 'LineString', coordinates: page.clipRing },
      },
      ...page.gcps.map((gcp): FeatureCollection['features'][0] => ({
        type: 'Feature',
        properties: { kind: 'gcp', gcpType: gcp.type },
        geometry: { type: 'Point', coordinates: [gcp.lon, gcp.lat] },
      })),
    ],
  };
}

/**
 * MapLibre map rendering a whole volume's pages, warped and clipped, via the
 * Allmaps WarpedMapLayer. Clicking a page selects it (bringing it to the front
 * of the stack); the selected page gets outline overlays. The map is created
 * once on mount; props are applied imperatively as they change.
 */
export function VolumeMap(props: VolumeMapProps) {
  const {
    annotation,
    pages,
    missingPages,
    truthPages,
    showMissing,
    selectedItemIndex,
    onSelectPage,
    opacity,
    awaitingView,
    onLoadResult,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const layerRef = useRef<WarpedMapLayer | null>(null);
  const [mapReady, setMapReady] = useState(false);
  // Whether the map has been fit to a loaded annotation.
  const [positioned, setPositioned] = useState(false);
  // Whether the map is shown. Latched true once there is something to show, so
  // it never reverts to hidden on a later volume/selection change.
  const [revealed, setRevealed] = useState(false);

  // Reveal once the map is fit to its annotation, or — when nothing is loading
  // and no annotation is present (the volume picker) — right away at the default view.
  useEffect(() => {
    if (revealed) return;
    if (mapReady && (positioned || (!annotation && !awaitingView))) {
      setRevealed(true);
    }
  }, [revealed, mapReady, positioned, annotation, awaitingView]);

  // Latest props for the click/hover handlers, which are installed once.
  const pagesRef = useRef(pages);
  const missingPagesRef = useRef(missingPages);
  const showMissingRef = useRef(showMissing);
  const onSelectPageRef = useRef(onSelectPage);
  const selectedRef = useRef(selectedItemIndex);
  useEffect(() => {
    pagesRef.current = pages;
    missingPagesRef.current = missingPages;
    showMissingRef.current = showMissing;
    onSelectPageRef.current = onSelectPage;
    selectedRef.current = selectedItemIndex;
  }, [pages, missingPages, showMissing, onSelectPage, selectedItemIndex]);

  // Allmaps map IDs indexed by annotation itemIndex, and the itemIndexes
  // brought to front so far (most recent last) so hit-testing can pick the
  // page that is visually on top of an overlap.
  const mapIdsRef = useRef<(string | null)[]>([]);
  const frontOrderRef = useRef<number[]>([]);

  // The map currently rendered without its clip mask (the selected page).
  const unmaskedMapIdRef = useRef<string | null>(null);

  // Keep the latest callback out of the annotation effect's dependencies so a
  // re-rendered parent doesn't re-add the annotation.
  const onLoadResultRef = useRef(onLoadResult);
  useEffect(() => {
    onLoadResultRef.current = onLoadResult;
  }, [onLoadResult]);

  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution:
              '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [-73.99, 40.7],
      zoom: 13,
      // WarpedMapLayer does not support pitch.
      maxPitch: 0,
    });
    mapRef.current = map;
    // Debug/e2e handle in the spirit of window.mapsnap: lets automated tests
    // drive the map, e.g. mapsnapVolumeMap.fire('click', {lngLat: {lng, lat}}).
    (window as { mapsnapVolumeMap?: maplibregl.Map }).mapsnapVolumeMap = map;
    map.addControl(
      new maplibregl.ScaleControl({ unit: 'imperial' }),
      'bottom-left',
    );
    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: false }),
      'top-left',
    );
    map.on('load', () => {
      const layer = new WarpedMapLayer();
      map.addLayer(layer);
      map.addSource('rmse-fills', {
        type: 'geojson',
        data: EMPTY_FEATURES,
      });
      map.addLayer({
        id: 'rmse-fill',
        type: 'fill',
        source: 'rmse-fills',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': RMSE_FILL_OPACITY,
          // No transition so the fill tracks the opacity slider instantly,
          // matching the warped maps (which animate: false below).
          'fill-opacity-transition': { duration: 0 },
          'fill-outline-color': ['get', 'color'],
        },
      });
      // Missing (un-fitted) pages, drawn at their truth footprint as translucent
      // white polygons with a dashed border to read as distinct from fit pages.
      map.addSource('missing-pages', {
        type: 'geojson',
        data: EMPTY_FEATURES,
      });
      map.addLayer({
        id: 'missing-pages-fill',
        type: 'fill',
        source: 'missing-pages',
        paint: { 'fill-color': '#ffffff', 'fill-opacity': 0.35 },
      });
      map.addLayer({
        id: 'missing-pages-outline',
        type: 'line',
        source: 'missing-pages',
        paint: {
          'line-color': '#333333',
          'line-width': 1.5,
          'line-dasharray': [3, 2],
        },
      });
      // Adjacency claim boxes: blue where the claimed neighbor claims back (mutual), red for
      // a one-sided claim. Drawn where each read sits on its page.
      map.addSource('adjacency-claims', {
        type: 'geojson',
        data: EMPTY_FEATURES,
      });
      // Claims on the selected page (or all of them when nothing is selected) draw filled with a
      // solid outline; `onSelectedPage` is set per feature in the source-populating effect.
      map.addLayer({
        id: 'adjacency-claims-fill',
        type: 'fill',
        source: 'adjacency-claims',
        paint: {
          'fill-color': ['case', ['get', 'mutual'], '#2563eb', '#d97706'],
          'fill-opacity': ['case', ['get', 'onSelectedPage'], 0.5, 0],
          'fill-outline-color': [
            'case',
            ['get', 'mutual'],
            '#1e3a8a',
            '#92400e',
          ],
        },
      });
      // Claims on other pages draw as a dashed outline only, reading as "underneath" the selection.
      map.addLayer({
        id: 'adjacency-claims-dashed',
        type: 'line',
        source: 'adjacency-claims',
        filter: ['!', ['get', 'onSelectedPage']],
        paint: {
          'line-color': ['case', ['get', 'mutual'], '#2563eb', '#d97706'],
          'line-width': 1.5,
          'line-dasharray': [2, 2],
        },
      });
      map.addSource('selected-page', {
        type: 'geojson',
        data: EMPTY_FEATURES,
      });
      // The selected page's truth box, drawn first so it sits under the generated outline:
      // where the human georeference places the same four image corners.
      map.addLayer({
        id: 'selected-page-truth',
        type: 'line',
        source: 'selected-page',
        filter: ['==', ['get', 'kind'], 'truth'],
        paint: { 'line-color': '#16a34a', 'line-width': 1.5 },
      });
      map.addLayer({
        id: 'selected-page-rect',
        type: 'line',
        source: 'selected-page',
        filter: ['==', ['get', 'kind'], 'rect'],
        paint: { 'line-color': '#000000', 'line-width': 2 },
      });
      map.addLayer({
        id: 'selected-page-clip',
        type: 'line',
        source: 'selected-page',
        filter: ['==', ['get', 'kind'], 'clip'],
        paint: {
          'line-color': '#000000',
          'line-width': 1.5,
          'line-dasharray': [2, 2],
        },
      });
      // GCP circles, colored like the georef view's intersection markers:
      // orange for real GCPs, grey for corner fallbacks.
      map.addLayer({
        id: 'selected-page-gcps',
        type: 'circle',
        source: 'selected-page',
        filter: ['==', ['get', 'kind'], 'gcp'],
        paint: {
          'circle-radius': 4,
          'circle-color': [
            'case',
            ['==', ['get', 'gcpType'], 'corner'],
            '#888888',
            'orange',
          ],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.5,
        },
      });
      layerRef.current = layer;
      setMapReady(true);
    });

    // The top-most page whose clipping polygon contains the point, or null.
    // The selected page renders unclipped, so its full rectangle counts as a
    // hit while it is selected. Otherwise the most-recently-selected page wins
    // an overlap, then the later-added page (Allmaps renders later additions
    // on top). Missing-page footprints are considered last (and only when
    // shown), so a fitted page always wins where the two overlap.
    function pageAtPoint(lng: number, lat: number): number | null {
      const selectedId = selectedRef.current;
      const selected =
        pagesRef.current.find((p) => p.itemIndex === selectedId) ??
        missingPagesRef.current.find((p) => p.itemIndex === selectedId);
      if (selected && pointInPolygon(lng, lat, selected.rectRing)) {
        return selected.itemIndex;
      }
      let best: number | null = null;
      let bestScore = -Infinity;
      for (const page of pagesRef.current) {
        if (!pointInPolygon(lng, lat, page.clipRing)) continue;
        const frontRank = frontOrderRef.current.lastIndexOf(page.itemIndex);
        const score = frontRank * 1e6 + page.itemIndex;
        if (score > bestScore) {
          bestScore = score;
          best = page.itemIndex;
        }
      }
      if (best !== null) return best;
      if (showMissingRef.current) {
        for (const page of missingPagesRef.current) {
          if (pointInPolygon(lng, lat, page.clipRing)) return page.itemIndex;
        }
      }
      return null;
    }

    map.on('click', (e) => {
      onSelectPageRef.current(pageAtPoint(e.lngLat.lng, e.lngLat.lat));
    });
    map.on('mousemove', (e) => {
      const hit = pageAtPoint(e.lngLat.lng, e.lngLat.lat) !== null;
      map.getCanvas().style.cursor = hit ? 'pointer' : '';
    });

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      setMapReady(false);
    };
  }, []);

  // Show the annotation's pages, replacing any previous volume's, and fit the view.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer || !mapReady) return;
    layer.clear();
    mapIdsRef.current = [];
    frontOrderRef.current = [];
    unmaskedMapIdRef.current = null;
    if (!annotation) return;
    const results = layer.addGeoreferenceAnnotation(annotation);
    mapIdsRef.current = results.map((r) => (typeof r === 'string' ? r : null));
    results.forEach((r, i) => {
      if (r instanceof Error) {
        const item = (annotation as { items?: { label?: unknown }[] }).items?.[
          i
        ];
        console.error(
          `addGeoreferenceAnnotation failed for item ${i}`,
          item?.label,
          r,
        );
      }
    });
    const failed = results.filter((r) => r instanceof Error).length;
    onLoadResultRef.current?.({ loaded: results.length - failed, failed });
    const bounds = layer.getBounds();
    if (bounds) map.fitBounds(bounds, { padding: 40, animate: false });
    setPositioned(true);
  }, [annotation, mapReady]);

  // Outline the selected page, bring it to the front of the stack, and remove
  // its clip mask so the whole sheet (margins and all) is visible.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer || !mapReady) return;
    const previousUnmasked = unmaskedMapIdRef.current;
    if (previousUnmasked) {
      layer.resetMapsOptions([previousUnmasked], ['applyMask'], {
        animate: false,
      });
      unmaskedMapIdRef.current = null;
    }
    const source = map.getSource<maplibregl.GeoJSONSource>('selected-page');
    // A missing page has no warped map to bring forward; its footprint is still
    // outlined here so selecting one (from the list or the map) shows where it is.
    const page =
      selectedItemIndex === null
        ? undefined
        : (pages.find((p) => p.itemIndex === selectedItemIndex) ??
          missingPages.find((p) => p.itemIndex === selectedItemIndex));
    if (!page) {
      source?.setData(EMPTY_FEATURES);
      return;
    }
    // A fitted page (non-negative id) shows the truth box(es) sharing its page key beneath its
    // outline; a missing page is itself truth, so it gets none.
    const truthRings =
      page.itemIndex >= 0
        ? truthPages
            .filter((truth) => truth.pageKey === page.pageKey)
            .map((truth) => truth.rectRing)
        : [];
    source?.setData(selectionFeatures(page, truthRings));
    const mapId = mapIdsRef.current[page.itemIndex];
    if (mapId) {
      layer.bringMapsToFront([mapId]);
      layer.setMapsOptions([mapId], { applyMask: false }, { animate: false });
      unmaskedMapIdRef.current = mapId;
      frontOrderRef.current.push(page.itemIndex);
    }
  }, [selectedItemIndex, pages, missingPages, truthPages, mapReady]);

  // Also depends on `annotation`: newly added maps start at full opacity, so
  // the current value must be reapplied after each volume load. Animation is
  // disabled so scrubbing the slider tracks instantly instead of queueing
  // 300ms transitions.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer || !mapReady) return;
    layer.setLayerOptions({ opacity }, { animate: false });
    // Fade the RMSE color fills along with the maps, proportional to their base.
    if (map.getLayer('rmse-fill')) {
      map.setPaintProperty(
        'rmse-fill',
        'fill-opacity',
        RMSE_FILL_OPACITY * opacity,
      );
    }
  }, [opacity, annotation, mapReady]);

  // RMSE color-coding: one translucent fill per page footprint when enabled.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const source = map.getSource<maplibregl.GeoJSONSource>('rmse-fills');
    if (!source) return;
    if (!props.pageColors) {
      source.setData(EMPTY_FEATURES);
      return;
    }
    const features = props.pages
      .filter((page) => props.pageColors?.has(page.itemIndex))
      .map((page): FeatureCollection['features'][0] => ({
        type: 'Feature',
        properties: { color: props.pageColors?.get(page.itemIndex) },
        geometry: { type: 'Polygon', coordinates: [page.clipRing] },
      }));
    source.setData({ type: 'FeatureCollection', features });
  }, [props.pageColors, props.pages, mapReady]);

  // Adjacency claim boxes, one polygon per claim. `mutual` drives the colour; `onSelectedPage`
  // (true for every claim when nothing is selected) drives filled-vs-dashed in the layer paint.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const source = map.getSource<maplibregl.GeoJSONSource>('adjacency-claims');
    if (!source) return;
    const { selectedStem } = props;
    source.setData({
      type: 'FeatureCollection',
      features: props.adjacencyClaims.map(
        (claim): FeatureCollection['features'][0] => ({
          type: 'Feature',
          properties: {
            mutual: claim.mutual,
            onSelectedPage:
              selectedStem === null || claim.stem === selectedStem,
          },
          geometry: { type: 'Polygon', coordinates: [claim.ring] },
        }),
      ),
    });
  }, [props.adjacencyClaims, props.selectedStem, mapReady]);

  // Missing-page footprints: one translucent white polygon per un-fitted page
  // at its truth clip ring, shown only when the toggle is on.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const source = map.getSource<maplibregl.GeoJSONSource>('missing-pages');
    if (!source) return;
    if (!showMissing) {
      source.setData(EMPTY_FEATURES);
      return;
    }
    source.setData({
      type: 'FeatureCollection',
      features: missingPages.map((page): FeatureCollection['features'][0] => ({
        type: 'Feature',
        properties: { itemIndex: page.itemIndex },
        geometry: { type: 'Polygon', coordinates: [page.clipRing] },
      })),
    });
  }, [missingPages, showMissing, mapReady]);

  return (
    <div
      id="map"
      ref={containerRef}
      style={{ visibility: revealed ? undefined : 'hidden' }}
    />
  );
}
