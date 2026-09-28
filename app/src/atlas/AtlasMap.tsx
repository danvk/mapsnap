/**
 * The atlas's one map: a country of dots that becomes a map of volumes.
 *
 * Both states share a single maplibre instance rather than swapping between
 * two, so zooming from the country into a town is one continuous movement.
 *
 * Zoomed out, each town is a dot. From FOOTPRINT_ZOOM in, the dots give way
 * to the footprints of the volumes in view: each town's newest coverage, a
 * patchwork rather than a stack of editions. A click on a footprint asks the
 * app for the newest volume there; only that volume's imagery is drawn, and
 * its footprint is outlined over it. Clicking inside it picks a sheet, and
 * clicking any other footprint moves on to that volume.
 */

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';

import { pointInPolygon } from '../geometry';
import type { LoadedVolume } from './annotations';
import { inMultiPolygon, type MultiPolygonCoords } from './footprints';
import type { Place } from './places';

/** Which sheet of which volume the pointer is over. */
export interface PageRef {
  item: string;
  itemIndex: number;
}

/** Where the map should move to: a box to fit, or a point and zoom. */
export type MapTarget =
  | { bounds: [number, number, number, number]; key: number }
  | { center: [number, number]; zoom: number; key: number };

/** Zoom from which volume footprints replace the town dots. */
export const FOOTPRINT_ZOOM = 10;
/** A footprint's shading when nothing is selected. */
const FILL_OPACITY = 0.12;

interface AtlasMapProps {
  places: Place[];
  /** What a dot's area means. */
  sizeBy: 'sheets' | 'volumes';
  /** The footprints to draw (see footprintFeatures). */
  footprints: GeoJSON.FeatureCollection;
  /** The selected volume's footprint, inside which a click picks a sheet. */
  selectedFootprint: MultiPolygonCoords | null;
  /** The selected volume's annotation, once fetched. */
  loaded: LoadedVolume | null;
  selectedPage: PageRef | null;
  /** Opacity of the warped sheets, in [0, 1]. */
  opacity: number;
  target: MapTarget | null;
  /** A town's dot was clicked. */
  onSelectPlace: (place: Place) => void;
  /** A spot outside the selected volume was clicked, from FOOTPRINT_ZOOM in. */
  onPickLocation: (lng: number, lat: number) => void;
  onSelectPage: (page: PageRef | null) => void;
  /** The view settled: [west, south, east, north] and zoom. */
  onViewChange: (
    bounds: [number, number, number, number],
    zoom: number,
  ) => void;
}

/**
 * A place's dot area tracks its size, so radius tracks the square root.
 *
 * Dots keep growing as you zoom toward a town, more slowly than the ground,
 * until the footprints take over.
 */
function radiusExpression(sizeBy: 'sheets' | 'volumes'): unknown {
  const magnitude = ['sqrt', ['max', ['get', sizeBy], 1]];
  const ceiling = sizeBy === 'sheets' ? 60 : 13; // sqrt(3600) and sqrt(170)
  const radii = (smallest: number, largest: number) => [
    'interpolate',
    ['linear'],
    magnitude,
    1,
    smallest,
    ceiling,
    largest,
  ];
  return [
    'interpolate',
    ['linear'],
    ['zoom'],
    3,
    radii(1.5, 11),
    7,
    radii(3, 26),
    FOOTPRINT_ZOOM,
    radii(4.5, 32),
  ];
}

/** GeoJSON for the dot layer: one point per town. */
function placesGeoJson(places: Place[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: places.map((place) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [place.lon, place.lat] },
      properties: {
        id: place.id,
        name: place.name,
        state: place.state,
        sheets: place.sheets,
        volumes: place.volumes,
        // Paint tells the two apart: a town the run has nothing for is still
        // a town the collection covers, and hiding it would misreport the
        // collection as smaller than it is.
        mirrored: place.mirrored > 0 ? 1 : 0,
      },
    })),
  };
}

const EMPTY: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: [],
};

export function AtlasMap(props: AtlasMapProps) {
  const {
    places,
    sizeBy,
    footprints,
    selectedFootprint,
    loaded,
    selectedPage,
    opacity,
    target,
    onSelectPlace,
    onPickLocation,
    onSelectPage,
    onViewChange,
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const layerRef = useRef<WarpedMapLayer | null>(null);
  const [ready, setReady] = useState(false);

  // The map's own listeners are registered once and read the latest props
  // through a ref; re-registering them every render would drop clicks between
  // removal and re-add.
  const latest = useRef({
    places,
    loaded,
    selectedFootprint,
    onSelectPlace,
    onPickLocation,
    onSelectPage,
    onViewChange,
  });
  useEffect(() => {
    latest.current = {
      places,
      loaded,
      selectedFootprint,
      onSelectPlace,
      onPickLocation,
      onSelectPage,
      onViewChange,
    };
  });

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
      center: [-97, 38.5],
      zoom: 3.6,
      maxPitch: 0,
    });
    mapRef.current = map;
    // Debug handle in the spirit of window.mapsnapVolumeMap: lets a console or
    // an automated test reach the map and the warped layer.
    (window as { mapsnapAtlas?: unknown }).mapsnapAtlas = { map };
    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: false }),
      'top-left',
    );
    map.addControl(new maplibregl.ScaleControl({ unit: 'imperial' }));

    const reportView = () => {
      const b = map.getBounds();
      latest.current.onViewChange(
        [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()],
        map.getZoom(),
      );
    };

    map.on('load', () => {
      const layer = new WarpedMapLayer();
      map.addLayer(layer);
      layerRef.current = layer;
      (window as { mapsnapAtlas?: { layer?: unknown } }).mapsnapAtlas!.layer =
        layer;

      map.addSource('footprints', {
        type: 'geojson',
        data: EMPTY,
        promoteId: 'item',
      });
      // Above the imagery, so a neighbouring volume stays one click away. The
      // selected volume has no fill -- its imagery is what should show -- only
      // a heavier outline around it.
      map.addLayer({
        id: 'footprint-fill',
        type: 'fill',
        source: 'footprints',
        minzoom: FOOTPRINT_ZOOM,
        paint: {
          'fill-color': '#2563eb',
          'fill-opacity': [
            'case',
            ['==', ['get', 'selected'], 1],
            0,
            ['boolean', ['feature-state', 'hover'], false],
            0.28,
            FILL_OPACITY,
          ],
        },
      });
      map.addLayer({
        id: 'footprint-line',
        type: 'line',
        source: 'footprints',
        minzoom: FOOTPRINT_ZOOM,
        paint: {
          'line-color': [
            'case',
            ['==', ['get', 'selected'], 1],
            '#f97316',
            '#2563eb',
          ],
          'line-width': ['case', ['==', ['get', 'selected'], 1], 2.5, 1],
          'line-opacity': 0.8,
        },
      });

      map.addSource('places', { type: 'geojson', data: EMPTY });
      map.addLayer({
        id: 'place-dots',
        type: 'circle',
        source: 'places',
        maxzoom: FOOTPRINT_ZOOM,
        paint: {
          'circle-color': [
            'case',
            ['==', ['get', 'mirrored'], 1],
            '#2563eb',
            '#9ca3af',
          ],
          'circle-opacity': 0.65,
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 0.5,
          'circle-stroke-opacity': 0.9,
        },
      });
      setReady(true);
      reportView();
    });

    map.on('moveend', reportView);

    // Below FOOTPRINT_ZOOM a click can only mean a dot. From it in, a click
    // inside the selected volume picks a sheet, and anywhere else asks for
    // the newest volume at that spot.
    map.on('click', (event) => {
      const { lng, lat } = event.lngLat;
      if (map.getZoom() < FOOTPRINT_ZOOM) {
        const hit = map.queryRenderedFeatures(event.point, {
          layers: ['place-dots'],
        })[0];
        const id = hit?.properties?.id as string | undefined;
        const place = id
          ? latest.current.places.find((entry) => entry.id === id)
          : undefined;
        if (place) latest.current.onSelectPlace(place);
        return;
      }
      const selected = latest.current.selectedFootprint;
      if (selected && inMultiPolygon(lng, lat, selected)) {
        latest.current.onSelectPage(pageAt(latest.current.loaded, lng, lat));
        return;
      }
      latest.current.onPickLocation(lng, lat);
    });

    let hovered: string | number | undefined;
    map.on('mousemove', (event) => {
      let pointer = false;
      if (map.getZoom() < FOOTPRINT_ZOOM) {
        pointer =
          map.queryRenderedFeatures(event.point, { layers: ['place-dots'] })
            .length > 0;
      } else {
        const hit = map.queryRenderedFeatures(event.point, {
          layers: ['footprint-fill'],
        });
        const next = hit.find((f) => f.properties?.selected !== 1)?.id;
        if (next !== hovered) {
          if (hovered !== undefined) {
            map.setFeatureState(
              { source: 'footprints', id: hovered },
              { hover: false },
            );
          }
          if (next !== undefined) {
            map.setFeatureState(
              { source: 'footprints', id: next },
              { hover: true },
            );
          }
          hovered = next;
        }
        pointer = hit.length > 0;
      }
      map.getCanvas().style.cursor = pointer ? 'pointer' : '';
    });

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    (map.getSource('places') as maplibregl.GeoJSONSource | null)?.setData(
      placesGeoJson(places),
    );
  }, [places, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    map.setPaintProperty(
      'place-dots',
      'circle-radius',
      radiusExpression(sizeBy) as never,
    );
  }, [sizeBy, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    (map.getSource('footprints') as maplibregl.GeoJSONSource | null)?.setData(
      footprints,
    );
  }, [footprints, ready]);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !ready) return;
    layer.setLayerOptions({ opacity }, { animate: false });
  }, [opacity, ready]);

  // With a volume selected, its neighbours keep their outlines and only a
  // trace of fill: shading over its sheets would tint the imagery on show.
  // They stay clickable, and brighten on hover.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    map.setPaintProperty('footprint-fill', 'fill-opacity', [
      'case',
      ['==', ['get', 'selected'], 1],
      0,
      ['boolean', ['feature-state', 'hover'], false],
      0.28,
      selectedFootprint ? 0.04 : FILL_OPACITY,
    ]);
  }, [selectedFootprint, ready]);

  // Draw the selected volume, and nothing else.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !ready) return;
    layer.clear();
    if (!loaded) return;
    const results = layer.addGeoreferenceAnnotation(loaded.annotation);
    const failed = results.filter((r) => r instanceof Error).length;
    if (failed > 0) {
      console.warn(`${loaded.volume.item}: ${failed} page(s) failed to add`);
    }
  }, [loaded, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !target) return;
    if ('bounds' in target) {
      map.fitBounds(target.bounds, { padding: 60, duration: 900, maxZoom: 16 });
    } else {
      map.flyTo({ center: target.center, zoom: target.zoom, duration: 900 });
    }
  }, [target, ready]);

  // Bring the selected sheet to the front and unmask it, so a page picked out
  // of a stack can actually be read.
  const frontedRef = useRef<string | null>(null);
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !ready) return;
    if (frontedRef.current) {
      try {
        layer.resetMapsOptions([frontedRef.current], ['applyMask'], {
          animate: false,
        });
      } catch {
        // The map went with its volume.
      }
      frontedRef.current = null;
    }
    if (!selectedPage || !loaded || loaded.volume.item !== selectedPage.item) {
      return;
    }
    const mapId = loaded.annotation.items?.[selectedPage.itemIndex]?.id;
    if (typeof mapId !== 'string') return;
    try {
      layer.bringMapsToFront([mapId]);
      layer.setMapsOptions([mapId], { applyMask: false }, { animate: false });
      frontedRef.current = mapId;
    } catch {
      // A page whose annotation Allmaps rejected has no map to raise.
    }
  }, [selectedPage, loaded, ready]);

  return <div ref={containerRef} className="atlas-map" />;
}

/**
 * The drawn sheet of a volume under a point, latest-added first.
 *
 * Allmaps draws later additions on top, which is what the eye sees, and so
 * what a click should pick.
 */
export function pageAt(
  loaded: LoadedVolume | null,
  lng: number,
  lat: number,
): PageRef | null {
  if (!loaded) return null;
  for (let j = loaded.pages.length - 1; j >= 0; j--) {
    const page = loaded.pages[j];
    if (page && pointInPolygon(lng, lat, page.clipRing)) {
      return { item: loaded.volume.item, itemIndex: page.itemIndex };
    }
  }
  return null;
}
