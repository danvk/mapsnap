/**
 * The atlas's one map: a country of dots that becomes a town of sheets.
 *
 * Both states share a single maplibre instance rather than swapping between
 * two. Zooming from the whole country into one town is the app's central
 * gesture, and it has to be a continuous movement -- a map that unmounts and
 * remounts loses that, and with it any sense of where the town was.
 *
 * The dots stay in the style throughout, fading out as the warped sheets come
 * up, so backing out of a town lands you where you started.
 */

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';

import { pointInPolygon } from '../geometry';
import type { LoadedVolume } from './annotations';
import type { Place } from './places';

/** Which sheet of which volume the pointer is over. */
export interface PageRef {
  item: string;
  itemIndex: number;
}

interface AtlasMapProps {
  places: Place[];
  /** What a dot's area means. */
  sizeBy: 'sheets' | 'volumes';
  /** The volumes to draw, already fetched. Empty while showing the country. */
  loaded: LoadedVolume[];
  selectedPlace: Place | null;
  selectedPage: PageRef | null;
  onSelectPlace: (place: Place) => void;
  onSelectPage: (page: PageRef | null) => void;
}

/** Zoom at which the dots have finished handing over to the sheets. */
const DOTS_FADE_ZOOM = 11;

/** A place's dot area tracks its size, so radius tracks the square root. */
function radiusExpression(sizeBy: 'sheets' | 'volumes'): unknown {
  const magnitude = ['sqrt', ['max', ['get', sizeBy], 1]];
  const ceiling = sizeBy === 'sheets' ? 60 : 13; // sqrt(3600) and sqrt(170)
  return [
    'interpolate',
    ['linear'],
    ['zoom'],
    3,
    ['interpolate', ['linear'], magnitude, 1, 1.5, ceiling, 11],
    7,
    ['interpolate', ['linear'], magnitude, 1, 3, ceiling, 26],
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

export function AtlasMap(props: AtlasMapProps) {
  const {
    places,
    sizeBy,
    loaded,
    selectedPlace,
    selectedPage,
    onSelectPlace,
    onSelectPage,
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const layerRef = useRef<WarpedMapLayer | null>(null);
  const [ready, setReady] = useState(false);

  // Handlers and hit-test data reach the map's own listeners through refs: the
  // listeners are registered once, and re-registering them on every render
  // would drop clicks between removal and re-add.
  const placesRef = useRef(places);
  const loadedRef = useRef(loaded);
  const onSelectPlaceRef = useRef(onSelectPlace);
  const onSelectPageRef = useRef(onSelectPage);
  useEffect(() => {
    loadedRef.current = loaded;
    onSelectPlaceRef.current = onSelectPlace;
    onSelectPageRef.current = onSelectPage;
  }, [loaded, onSelectPlace, onSelectPage]);

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

    map.on('load', () => {
      const layer = new WarpedMapLayer();
      map.addLayer(layer);
      layerRef.current = layer;
      (window as { mapsnapAtlas?: { layer?: unknown } }).mapsnapAtlas!.layer =
        layer;
      map.addSource('places', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      // Above the warped sheets, so a dot stays clickable over a town that is
      // already drawn -- that is how you get from one town to its neighbour.
      map.addLayer({
        id: 'place-dots',
        type: 'circle',
        source: 'places',
        paint: {
          'circle-color': [
            'case',
            ['==', ['get', 'mirrored'], 1],
            '#2563eb',
            '#9ca3af',
          ],
          'circle-opacity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            DOTS_FADE_ZOOM - 2,
            0.65,
            DOTS_FADE_ZOOM,
            0,
          ],
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 0.5,
          'circle-stroke-opacity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            DOTS_FADE_ZOOM - 2,
            0.9,
            DOTS_FADE_ZOOM,
            0,
          ],
        },
      });
      setReady(true);
    });

    // One click handler for both states: a dot if the pointer is on one, else
    // whichever drawn sheet is under it.
    map.on('click', (event) => {
      const hits = map.queryRenderedFeatures(event.point, {
        layers: ['place-dots'],
      });
      const id = hits[0]?.properties?.id as string | undefined;
      if (id && map.getZoom() < DOTS_FADE_ZOOM) {
        const place = placesRef.current.find((entry) => entry.id === id);
        if (place) {
          onSelectPlaceRef.current(place);
          return;
        }
      }
      onSelectPageRef.current(
        pageAt(loadedRef.current, event.lngLat.lng, event.lngLat.lat),
      );
    });
    map.on('mousemove', (event) => {
      const onDot =
        map.getZoom() < DOTS_FADE_ZOOM &&
        map.queryRenderedFeatures(event.point, { layers: ['place-dots'] })
          .length > 0;
      const onPage =
        !onDot &&
        pageAt(loadedRef.current, event.lngLat.lng, event.lngLat.lat) !== null;
      map.getCanvas().style.cursor = onDot || onPage ? 'pointer' : '';
    });

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    placesRef.current = places;
    const map = mapRef.current;
    if (!map || !ready) return;
    const source = map.getSource('places') as maplibregl.GeoJSONSource | null;
    source?.setData(placesGeoJson(places));
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

  // Draw the fetched volumes, and frame them. Every volume of a town-year goes
  // onto one layer, so sheets from different volumes of the same year overlap
  // as they do on paper.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer || !ready) return;
    layer.clear();
    if (loaded.length === 0) return;
    for (const entry of loaded) {
      const results = layer.addGeoreferenceAnnotation(entry.annotation);
      const failed = results.filter((r) => r instanceof Error).length;
      if (failed > 0) {
        console.warn(`${entry.volume.item}: ${failed} page(s) failed to add`);
      }
    }
    const bounds = layer.getBounds();
    if (bounds) {
      map.fitBounds(bounds as [number, number, number, number], {
        padding: 60,
        duration: 900,
      });
    }
  }, [loaded, ready]);

  // Fly to a town as soon as it is picked, without waiting on its annotations:
  // the movement is the feedback that the click registered.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !selectedPlace) return;
    map.flyTo({
      center: [selectedPlace.lon, selectedPlace.lat],
      zoom: Math.max(map.getZoom(), 13),
      duration: 900,
    });
  }, [selectedPlace, ready]);

  // Bring the selected sheet to the front and unmask it, so a page picked out
  // of a stack can actually be read.
  const frontedRef = useRef<string | null>(null);
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !ready) return;
    if (frontedRef.current) {
      layer.resetMapsOptions([frontedRef.current], ['applyMask'], {
        animate: false,
      });
      frontedRef.current = null;
    }
    if (!selectedPage) return;
    const entry = loaded.find((v) => v.volume.item === selectedPage.item);
    const mapId = entry?.annotation.items?.[selectedPage.itemIndex]?.id;
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
 * The drawn sheet under a point, latest-added first.
 *
 * Allmaps draws later additions on top, so the last volume added wins an
 * overlap -- which is what the eye sees, and so what a click should pick.
 */
export function pageAt(
  loaded: LoadedVolume[],
  lng: number,
  lat: number,
): PageRef | null {
  for (let i = loaded.length - 1; i >= 0; i--) {
    const entry = loaded[i];
    if (!entry) continue;
    for (let j = entry.pages.length - 1; j >= 0; j--) {
      const page = entry.pages[j];
      if (page && pointInPolygon(lng, lat, page.clipRing)) {
        return { item: entry.volume.item, itemIndex: page.itemIndex };
      }
    }
  }
  return null;
}
