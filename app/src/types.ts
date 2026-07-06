/** A street label position with optional fitted geographic coordinates. */
export interface Street {
  street: string;
  x: number;
  y: number;
  lat?: number;
  lon?: number;
  dir_x?: number;
  dir_y?: number;
  dir_lon?: number;
  dir_lat?: number;
  inlier?: boolean;
  /** Read by the key-map rectangle fallback vocab rather than the tighter radius vocab. */
  fallback?: boolean;
}

/** A street-crossing ground control point with image and geographic coordinates. */
export interface IntersectionPoint {
  label_a: string;
  label_b: string;
  x: number;
  y: number;
  lat: number;
  lon: number;
  inlier: boolean;
  initial?: boolean;
}

/** A single OCR text detection from detect_text.py. */
export interface Detection {
  polygon: [number, number][];
  text: string;
  confidence: number;
  angle: number;
  long_side: number;
  short_side: number;
  ignore?: boolean;
  hint?: boolean;
  /** Read by the key-map rectangle fallback vocab rather than the tighter radius vocab. */
  fallback?: boolean;
}

/** The key map's expected center and OCR/fit radius for a page (georef.json `keymap`). */
export interface KeymapLocation {
  /** Mean of the detections — misleading for split pages (blocks far apart); prefer centers. */
  lat: number;
  lon: number;
  radius_m: number;
  /** Every key-map detection of the page number, as [lon, lat] (one per split panel). */
  centers?: [number, number][];
  /** Segmented key-map block(s) for the page: world-space rings of [lon, lat] pairs. */
  regions?: [number, number][][];
}

/** Four image corners mapped to [lon, lat], in [nw, ne, se, sw] order. */
export type Corners = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

/** Georef-format JSON: streets, intersections, and optional precomputed corners. */
export interface GeorefData {
  width?: number;
  height?: number;
  corners?: Corners;
  streets?: Street[];
  intersections?: IntersectionPoint[];
  keymap?: KeymapLocation;
}

/** New-format streets.json: detection list wrapped with image metadata. */
export interface StreetsJsonData {
  width: number;
  height: number;
  timestamp: string;
  command: string[];
  streets: Detection[];
}

/** A single panel polygon ring (one [x, y] vertex per point, in pixel space). */
export type PanelPolygon = [number, number][];

/**
 * panels.json sidecar: panel polygons in reading order in the named image's
 * pixel frame. `panels[i - 1]` corresponds to the page's `__i.jpg` split.
 *
 * `labels`, when present, is parallel to `panels` and gives a display label for
 * each polygon (e.g. the page number for key-map page regions); the app shows it
 * instead of the positional index.
 */
export interface PanelsJsonData {
  image: string;
  width: number;
  height: number;
  manual?: boolean;
  panels: PanelPolygon[];
  labels?: string[];
}
