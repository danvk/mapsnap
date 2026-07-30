/**
 * How a new region is drawn.
 *
 * Both produce an ordinary ring; 'rectangle' just gets there in one drag, which suits
 * the many key-map blocks that are plain rectangles.
 */
export type DrawMode = 'polygon' | 'rectangle';

/** One hand-drawn page region: a closed ring in image-pixel space plus its page key. */
export interface RegionPolygon {
  /** Ring vertices as [x, y] in image pixels. Stored open; closed on save. */
  ring: [number, number][];
  /** Page key this region belongs to, e.g. "12" or "35A". Empty until entered. */
  text: string;
}

/**
 * A `<stem>.regions.panels.json` sidecar.
 *
 * Deliberately the same schema `page_regions` writes and `score_regions` reads, so a
 * hand-drawn sheet can be scored against a detected segmentation — or against the
 * OIM-projected footprints — with no conversion step.
 */
export interface RegionsJson {
  /** Filename of the labelled image within the volume's `raw/`, e.g. "p0.jpg". */
  image: string;
  width: number;
  height: number;
  /** Closed rings (first vertex repeated as the last), one per region. */
  panels: number[][][];
  /** Page key per panel, parallel to `panels`. */
  labels: string[];
}

/** A sidecar as written by the client; the server fills in `image`. */
export type RegionsWriteRequest = Omit<RegionsJson, 'image'>;

/** One detected page number from a `<stem>.keymap.json`, for auto-labelling. */
export interface KeymapDetection {
  text: string;
  /** Centre of the detection's box, in image pixels. */
  x: number;
  y: number;
}
