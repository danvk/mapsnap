import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { WarpedMapLayer } from '@allmaps/maplibre';

interface VolumeMapProps {
  /** Rewritten Georeference AnnotationPage to display, or null before load. */
  annotation: unknown;
  /** Warped-image opacity in [0, 1]. */
  opacity: number;
  /** Called with per-page add results whenever a new annotation is shown. */
  onLoadResult?: (result: { loaded: number; failed: number }) => void;
}

/**
 * MapLibre map rendering a whole volume's pages, warped and clipped, via the
 * Allmaps WarpedMapLayer. The map is created once on mount; the annotation and
 * opacity are applied imperatively as props change.
 */
export function VolumeMap(props: VolumeMapProps) {
  const { annotation, opacity, onLoadResult } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const layerRef = useRef<WarpedMapLayer | null>(null);
  const [mapReady, setMapReady] = useState(false);

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
      layerRef.current = layer;
      setMapReady(true);
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
    if (!annotation) return;
    const results = layer.addGeoreferenceAnnotation(annotation);
    const failed = results.filter((r) => r instanceof Error).length;
    onLoadResultRef.current?.({ loaded: results.length - failed, failed });
    const bounds = layer.getBounds();
    if (bounds) map.fitBounds(bounds, { padding: 40, animate: false });
  }, [annotation, mapReady]);

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
