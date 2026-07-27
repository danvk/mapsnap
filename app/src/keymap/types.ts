/** A single truth label: a point in image-pixel space and its text. */
export interface Label {
  x: number;
  y: number;
  text: string;
}

/** A <stem>.labels.json sidecar: truth labels for one key map image. */
export interface LabelsJson {
  /** Filename of the labeled image within the volume's `raw/`, e.g. "p0.jpg". */
  image: string;
  width: number;
  height: number;
  labels: Label[];
}

/**
 * A sidecar as written by the client: everything but `image`, which the server
 * fills in because only it knows which file the page's stem resolved to.
 */
export type LabelsWriteRequest = Omit<LabelsJson, 'image'>;

/** One key map page in the available-pages list, with its label count. */
export interface ImageInfo {
  /** Page id: volume path and stem, e.g. "queens_1950/vol2/p0". */
  name: string;
  /** Number of labels in the sidecar, or null if no sidecar exists yet. */
  labelCount: number | null;
  /** Whether the page has a keymap.json sidecar (vs. being listed for its truth). */
  hasKeymap: boolean;
}
