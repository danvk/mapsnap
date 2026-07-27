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
  /** Labels whose text has been entered (blue badge; hidden when zero). */
  withText: number;
  /** Labels still awaiting text (orange badge; hidden when zero). */
  withoutText: number;
  /** Whether the page has a keymap.json sidecar (vs. being listed for its truth).
   * Absent for lists where the notion does not apply (adjacency pages). */
  hasKeymap?: boolean;
}
