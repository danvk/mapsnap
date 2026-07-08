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
  /**
   * Adjacency-claim reciprocity (adjacency mode only): true renders blue (a mutual
   * neighbor), false amber (an unreciprocated claim); absent for non-claims.
   */
  mutual?: boolean;
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

/**
 * One seed-GCP-pair fit, precomputed by the Python fitter under `--debug`.
 *
 * `a`/`b` index into `intersections`. Everything the debugger needs to show this pair's
 * fit is precomputed here — image `corners`, which labels/intersections are inliers, the
 * score and error — so the frontend never re-runs any fit or scoring logic. `degenerate`
 * pairs (coincident/singular) carry only a/b.
 */
export interface GcpPairResult {
  a: number;
  b: number;
  corners?: Corners;
  score?: number;
  /** Indices into `streets` that are inliers under this pair's fit. */
  inlier_streets?: number[];
  /** Indices into `intersections` that are inliers under this pair's fit. */
  inlier_intersections?: number[];
  mean_error_m?: number | null;
  max_error_m?: number | null;
  degenerate?: boolean;
}

/** Georef-format JSON: streets, intersections, and optional precomputed corners. */
export interface GeorefData {
  width?: number;
  height?: number;
  corners?: Corners;
  streets?: Street[];
  intersections?: IntersectionPoint[];
  keymap?: KeymapLocation;
  /** This page's human (OIM truth) footprint(s): world-space [lon, lat] rings, one per split. */
  truth?: [number, number][][];
  /** Per-seed-pair fits for interactive RANSAC exploration (present only with `--debug`). */
  gcp_pairs?: GcpPairResult[];
  /** True for a `.georef-nofit.json`: the pipeline georeferenced no fit for this page. */
  nofit?: boolean;
}

/** New-format streets.json: detection list wrapped with image metadata. */
export interface StreetsJsonData {
  width: number;
  height: number;
  timestamp: string;
  command: string[];
  streets: Detection[];
}

/** One digit read from page_adjacency.py (a valid page number found on a page). */
export interface AdjacencyDetection {
  number: number;
  text: string;
  confidence: number;
  polygon: [number, number][];
  height: number;
  x_frac: number;
  y_frac: number;
  /** Page edge the detection is near ("N"/"E"/"S"/"W"), a compass corner ("NE"), or "center". */
  edge: string;
  /** Passed the adjacency-claim filters (edge band, min height, not the page's own number). */
  claim: boolean;
}

/** One page's entry in adjacency.json. */
export interface AdjacencyPage {
  number: number | null;
  /** Dimensions of the scanned image, for rescaling polygons to the loaded image. */
  width?: number;
  height?: number;
  detections: AdjacencyDetection[];
}

/** adjacency.json: per-page sheet-number detections plus the mutual-edge adjacency graph. */
export interface AdjacencyData {
  pages: Record<string, AdjacencyPage>;
  /** Reciprocated adjacency edges as [stemA, stemB] pairs. */
  adjacency: [string, string][];
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
