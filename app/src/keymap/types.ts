/** A single truth label: a point in image-pixel space and its text. */
export interface Label {
  x: number;
  y: number;
  text: string;
}

/** A <stem>.labels.json sidecar: truth labels for one key map image. */
export interface LabelsJson {
  image: string;
  width: number;
  height: number;
  labels: Label[];
}

/** One key map image in the available-images list, with its label count. */
export interface ImageInfo {
  name: string;
  /** Number of labels in the sidecar, or null if no sidecar exists yet. */
  labelCount: number | null;
}
