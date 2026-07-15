import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';
import type { FeatureCollection } from 'geojson';
import { pointInPolygon } from '../geometry';
import type { PageGeo } from '../iiif/pages';

interface VolumeMapProps {
  /** Rewritten Georeference AnnotationPage to display, or null before load. */
  annotation: unknown;
  /** Derived page geometry for hit-testing and outlines. */
  pages: PageGeo[];
  /** itemIndex of the selected page, or null for no selection. */
  selectedItemIndex: number | null;
  /** Called with the clicked page's itemIndex, or null for empty space. */
  onSelectPage: (itemIndex: number | null) => void;
  /** Warped-image opacity in [0, 1]. */
  opacity: number;
  /** Called with per-page add results whenever a new annotation is shown. */
  onLoadResult?: (result: { loaded: number; failed: number }) => void;
}

const EMPTY_FEATURES: FeatureCollection = {
  type: 'FeatureCollection',
  features: [],
};

// Overlay features for a selected page: the full image rectangle (solid), the
// clipping polygon (dashed), and its GCPs (circles), distinguished by `kind`.
function selectionFeatures(page: PageGeo): FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [
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
    selectedItemIndex,
    onSelectPage,
    opacity,
    onLoadResult,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const layerRef = useRef<WarpedMapLayer | null>(null);
  const [mapReady, setMapReady] = useState(false);

  // Latest props for the click/hover handlers, which are installed once.
  const pagesRef = useRef(pages);
  const onSelectPageRef = useRef(onSelectPage);
  const selectedRef = useRef(selectedItemIndex);
  useEffect(() => {
    pagesRef.current = pages;
    onSelectPageRef.current = onSelectPage;
    selectedRef.current = selectedItemIndex;
  }, [pages, onSelectPage, selectedItemIndex]);

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
      map.addSource('selected-page', {
        type: 'geojson',
        data: EMPTY_FEATURES,
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
    // on top).
    function pageAtPoint(lng: number, lat: number): number | null {
      const selected = pagesRef.current.find(
        (p) => p.itemIndex === selectedRef.current,
      );
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
      return best;
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
    const failed = results.filter((r) => r instanceof Error).length;
    onLoadResultRef.current?.({ loaded: results.length - failed, failed });
    const bounds = layer.getBounds();
    if (bounds) map.fitBounds(bounds, { padding: 40, animate: false });
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
    const page =
      selectedItemIndex === null
        ? undefined
        : pages.find((p) => p.itemIndex === selectedItemIndex);
    if (!page) {
      source?.setData(EMPTY_FEATURES);
      return;
    }
    source?.setData(selectionFeatures(page));
    const mapId = mapIdsRef.current[page.itemIndex];
    if (mapId) {
      layer.bringMapsToFront([mapId]);
      layer.setMapsOptions([mapId], { applyMask: false }, { animate: false });
      unmaskedMapIdRef.current = mapId;
      frontOrderRef.current.push(page.itemIndex);
    }
  }, [selectedItemIndex, pages, mapReady]);

  // Also depends on `annotation`: newly added maps start at full opacity, so
  // the current value must be reapplied after each volume load. Animation is
  // disabled so scrubbing the slider tracks instantly instead of queueing
  // 300ms transitions.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !mapReady) return;
    layer.setLayerOptions({ opacity }, { animate: false });
  }, [opacity, annotation, mapReady]);

  return <div id="map" ref={containerRef} />;
}
