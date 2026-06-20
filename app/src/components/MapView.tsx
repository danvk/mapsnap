import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Corners, IntersectionPoint, Street } from '../types';
import { distanceMiles } from '../geometry';

interface MapViewProps {
  streets: Street[];
  intersections: IntersectionPoint[];
  corners: Corners | null;
  imageSrc: string;
  /** Warped-image opacity in [0, 1]. */
  opacity: number;
  showLabels: boolean;
  showIntersections: boolean;
  colorByInlier: boolean;
}

// Circle color expression for street labels, optionally split by inlier status.
function streetCircleColor(
  colorByInlier: boolean,
): maplibregl.ExpressionSpecification | string {
  return colorByInlier
    ? ([
        'case',
        ['get', 'inlier'],
        'orange',
        '#888888',
      ] as maplibregl.ExpressionSpecification)
    : '#ff0000';
}

// Text/line color for street labels, optionally split by inlier status.
function streetTextColor(colorByInlier: boolean): string {
  return colorByInlier ? 'orange' : '#ff0000';
}

/**
 * MapLibre map showing the warped image overlay plus street labels and
 * intersection GCPs. The map is created once on mount; layers are updated
 * imperatively as props change.
 */
export function MapView(props: MapViewProps) {
  const {
    streets,
    intersections,
    corners,
    imageSrc,
    opacity,
    showLabels,
    showIntersections,
    colorByInlier,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const lastWarpedUrlRef = useRef('');
  const [mapReady, setMapReady] = useState(false);

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
    });
    mapRef.current = map;
    map.on('load', () => setMapReady(true));
    return () => {
      map.remove();
      mapRef.current = null;
      setMapReady(false);
    };
  }, []);

  // Warp the image onto the map and fit the view when the image changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady || !corners || !imageSrc) return;

    const existing = map.getSource('warped') as
      | maplibregl.ImageSource
      | undefined;
    if (existing) {
      existing.updateImage({ url: imageSrc, coordinates: corners });
    } else {
      map.addSource('warped', {
        type: 'image',
        url: imageSrc,
        coordinates: corners,
      });
      map.addLayer({
        id: 'warped',
        type: 'raster',
        source: 'warped',
        paint: { 'raster-opacity': opacity },
      });
    }

    if (imageSrc !== lastWarpedUrlRef.current) {
      lastWarpedUrlRef.current = imageSrc;
      const lons = corners.map((c) => c[0]);
      const lats = corners.map((c) => c[1]);
      const minLon = Math.min(...lons);
      const maxLon = Math.max(...lons);
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const newCenterLon = (minLon + maxLon) / 2;
      const newCenterLat = (minLat + maxLat) / 2;
      const currentCenter = map.getCenter();
      const dist = distanceMiles(
        currentCenter.lng,
        currentCenter.lat,
        newCenterLon,
        newCenterLat,
      );
      map.fitBounds(
        [
          [minLon, minLat],
          [maxLon, maxLat],
        ],
        { padding: 40, maxZoom: 17, animate: dist <= 10 },
      );
    }
  }, [mapReady, corners, imageSrc, opacity]);

  // Apply opacity changes without refitting the view.
  useEffect(() => {
    const map = mapRef.current;
    if (map && mapReady && map.getLayer('warped')) {
      map.setPaintProperty('warped', 'raster-opacity', opacity);
    }
  }, [mapReady, opacity]);

  // Render street label positions and direction vectors.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;

    const geo = streets.filter(
      (s) => s.lat !== undefined && s.lon !== undefined,
    );

    const pointsGeojson: GeoJSON.FeatureCollection = {
      type: 'FeatureCollection',
      features: geo.map((s) => ({
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: [s.lon!, s.lat!] },
        properties: { label: s.street, inlier: s.inlier ?? true },
      })),
    };

    // Direction arrows: extend ±arrowHalf degrees from the label position.
    const arrowHalf = 0.0005;
    const linesGeojson: GeoJSON.FeatureCollection = {
      type: 'FeatureCollection',
      features: geo
        .filter((s) => s.dir_lon !== undefined && s.dir_lat !== undefined)
        .map((s) => ({
          type: 'Feature' as const,
          geometry: {
            type: 'LineString' as const,
            coordinates: [
              [
                s.lon! - s.dir_lon! * arrowHalf,
                s.lat! - s.dir_lat! * arrowHalf,
              ],
              [
                s.lon! + s.dir_lon! * arrowHalf,
                s.lat! + s.dir_lat! * arrowHalf,
              ],
            ],
          },
          properties: {},
        })),
    };

    const existingPts = map.getSource('street-labels') as
      | maplibregl.GeoJSONSource
      | undefined;
    if (existingPts) {
      existingPts.setData(pointsGeojson);
    } else {
      map.addSource('street-labels', { type: 'geojson', data: pointsGeojson });
      map.addLayer({
        id: 'street-labels-circle',
        type: 'circle',
        source: 'street-labels',
        paint: {
          'circle-radius': 5,
          'circle-color': streetCircleColor(colorByInlier),
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.5,
        },
      });
      map.addLayer({
        id: 'street-labels-text',
        type: 'symbol',
        source: 'street-labels',
        layout: {
          'text-field': ['get', 'label'],
          'text-font': ['Open Sans Regular'],
          'text-size': 10,
          'text-offset': [0, 1.2],
          'text-anchor': 'top',
        },
        paint: {
          'text-color': streetTextColor(colorByInlier),
          'text-halo-color': '#ffffff',
          'text-halo-width': 1.5,
        },
      });
    }

    const existingLines = map.getSource('street-vectors') as
      | maplibregl.GeoJSONSource
      | undefined;
    if (existingLines) {
      existingLines.setData(linesGeojson);
    } else {
      map.addSource('street-vectors', { type: 'geojson', data: linesGeojson });
      map.addLayer({
        id: 'street-vectors-line',
        type: 'line',
        source: 'street-vectors',
        paint: {
          'line-color': streetTextColor(colorByInlier),
          'line-width': 2,
          'line-opacity': 0.9,
        },
      });
    }

    if (map.getLayer('street-labels-circle'))
      map.setPaintProperty(
        'street-labels-circle',
        'circle-color',
        streetCircleColor(colorByInlier),
      );
    if (map.getLayer('street-labels-text'))
      map.setPaintProperty(
        'street-labels-text',
        'text-color',
        streetTextColor(colorByInlier),
      );
    if (map.getLayer('street-vectors-line'))
      map.setPaintProperty(
        'street-vectors-line',
        'line-color',
        streetTextColor(colorByInlier),
      );

    const visible = showLabels ? 'visible' : 'none';
    for (const id of [
      'street-labels-circle',
      'street-labels-text',
      'street-vectors-line',
    ]) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
    }
  }, [mapReady, streets, showLabels, colorByInlier]);

  // Render intersection GCPs.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;

    const colorExpr = [
      'case',
      ['get', 'initial'],
      '#0080ff',
      ['get', 'inlier'],
      '#ff0000',
      '#e6b800',
    ] as maplibregl.ExpressionSpecification;

    const geojson: GeoJSON.FeatureCollection = {
      type: 'FeatureCollection',
      features: intersections.map((ix) => ({
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: [ix.lon, ix.lat] },
        properties: {
          label: `${ix.label_a}\n${ix.label_b}`,
          inlier: ix.inlier,
          initial: ix.initial ?? false,
        },
      })),
    };

    const existing = map.getSource('intersections') as
      | maplibregl.GeoJSONSource
      | undefined;
    if (existing) {
      existing.setData(geojson);
    } else {
      map.addSource('intersections', { type: 'geojson', data: geojson });
      map.addLayer({
        id: 'intersections-circle',
        type: 'circle',
        source: 'intersections',
        paint: {
          'circle-radius': 7,
          'circle-color': colorExpr,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
        },
      });
      map.addLayer({
        id: 'intersections-text',
        type: 'symbol',
        source: 'intersections',
        layout: {
          'text-field': ['get', 'label'],
          'text-font': ['Open Sans Regular'],
          'text-size': 10,
          'text-offset': [1.4, 0],
          'text-anchor': 'left',
          'text-justify': 'left',
        },
        paint: {
          'text-color': colorExpr,
          'text-halo-color': '#ffffff',
          'text-halo-width': 2,
        },
      });
    }

    const visible = showIntersections ? 'visible' : 'none';
    for (const id of ['intersections-circle', 'intersections-text']) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
    }
  }, [mapReady, intersections, showIntersections]);

  return <div id="map" ref={containerRef} />;
}
