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
/** A measured CIELAB colour, as recorded by `detect_text.region_color`. */
export interface LabColor {
  /** sRGB hex ('#rrggbb'), for display. */
  color: string;
  /** Degrees: 0 = red, 90 = yellow, 180 = green, 270 = blue. */
  hue: number;
  /** Distance from neutral grey; 0 is unsaturated. */
  chroma: number;
}

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
   * The background under this detection, set only when it is more saturated than the page's
   * paper — i.e. the label is printed on a coloured building fill rather than on the paper
   * where street names belong. Absent for the ordinary on-paper case.
   */
  background?: LabColor;
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
  /**
   * Image-relative edge the detection is near — "T"op/"B"ottom/"L"eft/"R"ight, a corner
   * ("TL"/"TR"/"BL"/"BR"), or "center". Not compass directions: the page's world
   * orientation is unknown at this stage.
   */
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

/**
 * A single CRAFT detection box, mapped into the original image's pixel space and
 * tagged with the rotation pass that found it (before any text recognition).
 */
export interface Box {
  polygon: [number, number][];
  /** Rotation pass that produced the box, in degrees: 0, 90, or 270. */
  angle: number;
  /** Length of the box's longer side, in image pixels. */
  long_side: number;
  /** Length of the box's shorter side, in image pixels. */
  short_side: number;
}

/**
 * One rotation pass's raw boxes in boxes.json, in the rotated image's coordinate
 * frame (CRAFT is run on the image rotated by `angle`).
 */
export interface BoxAngleGroup {
  angle: number;
  /** Axis-aligned boxes as [x_min, x_max, y_min, y_max] in the rotated frame. */
  horizontal_list: [number, number, number, number][];
  /** Free (quadrilateral) boxes as [x, y] corner lists in the rotated frame. */
  free_list: [number, number][][];
}

/**
 * boxes.json sidecar: the raw CRAFT detection boxes from `mapsnap ocr`, grouped by
 * rotation pass, before recognition assigns any text.
 */
export interface BoxesJsonData {
  width: number;
  height: number;
  timestamp: string;
  command: string[];
  boxes: BoxAngleGroup[];
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
