import type { PanelPolygon } from '../types';

/** One arm's split of a page, scored against the truth by split_review.py. */
export interface ArmSplit {
  /** Open rings in the page image's pixel frame; one panel = left whole. */
  panels: PanelPolygon[];
  /** Matched IoU against the truth panels (score_splits_oim.py's metric). */
  iou: number;
}

/** One page of a split review. */
export interface ReviewPage {
  /** Image stem, unique within the review, e.g. "sanborn01778_005__p1". */
  name: string;
  /** Page image, relative to the review.json. */
  image: string;
  /** Volume title, e.g. "Champaign, Ill. | 1909"; may be empty. */
  title: string;
  item: string;
  page: string;
  /** Whether OIM's volunteers split the page. */
  label: 'split' | 'unsplit';
  width: number;
  height: number;
  /** OIM's panels; the whole page for an unsplit one. */
  truth: PanelPolygon[];
  /** Arm A's split, then arm B's. */
  arms: [ArmSplit, ArmSplit];
}

/**
 * A `review.json`, as scripts/split_review.py writes it.
 *
 * Self-contained: page images sit beside it, so the review is just a directory
 * under data/ that the dev server already serves.
 */
export interface SplitReview {
  title: string;
  /** Labels of arm A and arm B. */
  arms: [string, string];
  pages: ReviewPage[];
}
