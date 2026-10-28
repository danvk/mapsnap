/**
 * The atlas's one map: a country of dots that becomes a map of volumes.
 *
 * Both states share a single maplibre instance rather than swapping between
 * two, so zooming from the country into a town is one continuous movement.
 *
 * Zoomed out, each town is a dot. From FOOTPRINT_ZOOM in, the dots give way
 * to the volumes in view: one shape per volume of a town, a patchwork rather
 * than a stack of editions. A town the run placed nothing of keeps its dot,
 * since that is the only way in to its unplaced sheets. A click on a shape
 * selects that volume; only the selected edition's imagery is drawn, and the
 * volume is outlined around it. Clicking a drawn sheet picks it, and clicking
 * any other shape moves on to that volume.
 *
 * Layer order, bottom to top: the volumes' shapes, so they never tint the
 * imagery; a placeholder for each sheet of the selected edition until its
 * first tile arrives; the imagery; the selected volume's outline; the dots.
 */

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';

import { pointInPolygon } from '../geometry';
import type { LoadedVolume } from './annotations';
import {
  inMultiPolygon,
  type Bounds,
  type MultiPolygonCoords,
} from './footprints';
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
/** A volume's shading. */
const FILL_OPACITY = 0.12;
/** The two ends of a loading placeholder's pulse. */
const PLACEHOLDER_OPACITY: [number, number] = [0.2, 0.5];
const PLACEHOLDER_PULSE_MS = 700;

interface AtlasMapProps {
  places: Place[];
  /** The towns with a shape on the map; the others keep a dot when zoomed in. */
  townsWithShapes: ReadonlySet<string>;
  /** What a dot's area means. */
  sizeBy: 'sheets' | 'volumes';
  /** The footprints to draw (see footprintFeatures). */
  footprints: GeoJSON.FeatureCollection;
  /** Whether a volume is selected, which dims the other shapes. */
  hasSelection: boolean;
  /** The selected edition's footprint, inside which a click picks a sheet. */
  selectedFootprint: MultiPolygonCoords | null;
  /** The selected edition's annotation, once fetched. */
  loaded: LoadedVolume | null;
  selectedPage: PageRef | null;
  /** Opacity of the warped sheets, in [0, 1]. */
  opacity: number;
  target: MapTarget | null;
  /** A town's dot was clicked. */
  onSelectPlace: (place: Place) => void;
  /** A spot outside the selected edition was clicked, from FOOTPRINT_ZOOM in. */
  onPickLocation: (lng: number, lat: number) => void;
  /**
   * A sheet was clicked, or a spot inside the selected edition with no sheet
   * (null), or a spot with no volume at all (null, and no point).
   */
  onSelectPage: (page: PageRef | null, point: [number, number] | null) => void;
  /** The view settled: [west, south, east, north] and zoom. */
  onViewChange: (bounds: Bounds, zoom: number) => void;
}

/** The dot layers: the country's, and the towns with nothing placed, zoomed in. */
const DOT_LAYERS = ['place-dots', 'place-dots-unplaced'];

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

/** GeoJSON for the dot layers: one point per town. */
function placesGeoJson(
  places: Place[],
  townsWithShapes: ReadonlySet<string>,
): GeoJSON.FeatureCollection {
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
        placed: townsWithShapes.has(place.id) ? 1 : 0,
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
    townsWithShapes,
    sizeBy,
    footprints,
    hasSelection,
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
  // The volume shape under the pointer, highlighted while nothing is selected.
  const hoveredRef = useRef<string | number | undefined>(undefined);
  // The selected edition's maps whose imagery has not arrived yet.
  const pendingRef = useRef(new Set<string>());

  // The map's own listeners are registered once and read the latest props
  // through a ref; re-registering them every render would drop clicks between
  // removal and re-add.
  const latest = useRef({
    places,
    loaded,
    hasSelection,
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
      hasSelection,
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
      map.addSource('footprints', {
        type: 'geojson',
        data: EMPTY,
        promoteId: 'key',
      });
      // Under the imagery, so a neighboring volume never tints the sheets on
      // show; it stays clickable wherever the imagery leaves it in view. The
      // selected volume has no fill, only its outline, drawn above.
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
        filter: ['==', ['get', 'selected'], 0],
        paint: {
          'line-color': '#2563eb',
          'line-width': 1,
          'line-opacity': 0.8,
        },
      });

      // A sheet whose imagery is on its way: its masked outline, pulsing,
      // until the first tile lands on top of it.
      map.addSource('placeholders', { type: 'geojson', data: EMPTY });
      map.addLayer({
        id: 'placeholder-fill',
        type: 'fill',
        source: 'placeholders',
        paint: {
          'fill-color': '#9ca3af',
          'fill-opacity': PLACEHOLDER_OPACITY[0],
          'fill-opacity-transition': { duration: PLACEHOLDER_PULSE_MS },
        },
      });
      map.addLayer({
        id: 'placeholder-line',
        type: 'line',
        source: 'placeholders',
        paint: {
          'line-color': '#6b7280',
          'line-width': 1,
          'line-dasharray': [2, 2],
        },
      });

      const layer = new WarpedMapLayer();
      map.addLayer(layer);
      layerRef.current = layer;
      (window as { mapsnapAtlas?: { layer?: unknown } }).mapsnapAtlas!.layer =
        layer;

      map.addLayer({
        id: 'footprint-selected',
        type: 'line',
        source: 'footprints',
        filter: ['==', ['get', 'selected'], 1],
        paint: {
          'line-color': '#f97316',
          'line-width': 2.5,
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
      // Zoomed in, a town with nothing placed has no shape to click, so its
      // dot stays: a ring, to tell it from the shapes' fill.
      map.addLayer({
        id: 'place-dots-unplaced',
        type: 'circle',
        source: 'places',
        minzoom: FOOTPRINT_ZOOM,
        filter: [
          'all',
          ['==', ['get', 'placed'], 0],
          ['==', ['get', 'mirrored'], 1],
        ],
        paint: {
          'circle-radius': 8,
          'circle-color': '#fff',
          'circle-opacity': 0.85,
          'circle-stroke-color': '#2563eb',
          'circle-stroke-width': 2.5,
        },
      });
      setReady(true);
      reportView();
    });

    map.on('moveend', reportView);

    // A dot is a town. Otherwise a click on a drawn sheet picks it, and one
    // inside the selected edition between its sheets picks none. Elsewhere,
    // from FOOTPRINT_ZOOM in, it asks for the volume at that spot.
    map.on('click', (event) => {
      const { lng, lat } = event.lngLat;
      const dot = map.queryRenderedFeatures(event.point, {
        layers: DOT_LAYERS,
      })[0];
      const id = dot?.properties?.id as string | undefined;
      const place = id
        ? latest.current.places.find((entry) => entry.id === id)
        : undefined;
      if (place) {
        latest.current.onSelectPlace(place);
        return;
      }
      const page = pageAt(latest.current.loaded, lng, lat);
      const selected = latest.current.selectedFootprint;
      if (page || (selected && inMultiPolygon(lng, lat, selected))) {
        latest.current.onSelectPage(page, [lng, lat]);
        return;
      }
      if (map.getZoom() < FOOTPRINT_ZOOM) {
        latest.current.onSelectPage(null, null);
        return;
      }
      latest.current.onPickLocation(lng, lat);
    });

    // A dot names its town on hover: zoomed out, there is no other way to
    // tell which town a dot is before clicking it.
    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      className: 'atlas-dot-popup',
      offset: 10,
    });
    map.on('mousemove', (event) => {
      let pointer = false;
      const dot = map.queryRenderedFeatures(event.point, {
        layers: DOT_LAYERS,
      })[0];
      if (dot) {
        const { name, state, volumes } = dot.properties as {
          name: string;
          state: string;
          volumes: number;
        };
        popup
          .setLngLat(event.lngLat)
          .setText(
            `${name}, ${state} · ${volumes} volume${volumes === 1 ? '' : 's'}`,
          )
          .addTo(map);
      } else {
        popup.remove();
      }
      if (dot || map.getZoom() < FOOTPRINT_ZOOM) {
        pointer = dot !== undefined;
      } else {
        const hit = map.queryRenderedFeatures(event.point, {
          layers: ['footprint-fill'],
        });
        // With an edition on show, a highlight flickering across the volumes
        // around it would only distract from the imagery.
        const next = latest.current.hasSelection
          ? undefined
          : hit.find((f) => f.properties?.selected !== 1)?.id;
        setHovered(map, hoveredRef, next);
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
      placesGeoJson(places, townsWithShapes),
    );
  }, [places, townsWithShapes, ready]);

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
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer || !ready) return;
    layer.setLayerOptions({ opacity }, { animate: false });
    // Hidden imagery (the space bar, or the slider at 0) hides its
    // placeholders too: the point is to see the map underneath.
    for (const id of ['placeholder-fill', 'placeholder-line']) {
      map.setLayoutProperty(id, 'visibility', opacity > 0 ? 'visible' : 'none');
    }
  }, [opacity, ready]);

  // A placeholder for each sheet of the selected edition, until its imagery
  // starts to arrive: Allmaps reports each tile it loads, by map.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const source = map.getSource('placeholders') as maplibregl.GeoJSONSource;
    const pending = pendingRef.current;
    pending.clear();
    const features: GeoJSON.Feature[] = [];
    for (const page of loaded?.pages ?? []) {
      const mapId = loaded?.annotation.items?.[page.itemIndex]?.id;
      if (typeof mapId !== 'string') continue;
      pending.add(mapId);
      features.push({
        type: 'Feature',
        id: page.itemIndex,
        geometry: { type: 'Polygon', coordinates: [page.clipRing] },
        properties: { mapId },
      });
    }
    let frame = 0;
    const redraw = () => {
      frame = 0;
      source.setData({
        type: 'FeatureCollection',
        features: features.filter((f) => pending.has(f.properties?.mapId)),
      });
    };
    redraw();
    const onTile = (event: { mapIds?: string[] }) => {
      let changed = false;
      for (const mapId of event.mapIds ?? []) changed ||= pending.delete(mapId);
      if (changed && !frame) frame = requestAnimationFrame(redraw);
    };
    map.on('maptileloaded', onTile);
    // The pulse: back and forth between two opacities while anything waits.
    let bright = false;
    const pulse = window.setInterval(() => {
      if (pending.size === 0) return;
      bright = !bright;
      map.setPaintProperty(
        'placeholder-fill',
        'fill-opacity',
        PLACEHOLDER_OPACITY[bright ? 1 : 0],
      );
    }, PLACEHOLDER_PULSE_MS);
    return () => {
      map.off('maptileloaded', onTile);
      window.clearInterval(pulse);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [loaded, ready]);

  // Selecting something ends any highlight; hovering adds none until it is closed.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (hasSelection) setHovered(map, hoveredRef, undefined);
  }, [hasSelection, ready]);

  // Draw the selected edition, and nothing else.
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

/** Move the hover highlight to another volume shape, or clear it. */
function setHovered(
  map: maplibregl.Map,
  hoveredRef: { current: string | number | undefined },
  next: string | number | undefined,
): void {
  if (next === hoveredRef.current) return;
  if (hoveredRef.current !== undefined) {
    map.setFeatureState(
      { source: 'footprints', id: hoveredRef.current },
      { hover: false },
    );
  }
  if (next !== undefined) {
    map.setFeatureState({ source: 'footprints', id: next }, { hover: true });
  }
  hoveredRef.current = next;
}

/**
 * The drawn sheet of an edition under a point, latest-added first.
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
