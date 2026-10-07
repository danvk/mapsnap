import type { ReviewPage, SplitReview } from './types';

/** An arm's result on one page, as the review shows it. */
export interface ArmOutcome {
  panels: number;
  truthPanels: number;
  /** Whether the arm produced the truth's panel count. */
  right: boolean;
  iou: number;
}

/** Arm `arm` (0 = A, 1 = B) on a page, against its truth. */
export function armOutcome(page: ReviewPage, arm: 0 | 1): ArmOutcome {
  const split = page.arms[arm];
  return {
    panels: split.panels.length,
    truthPanels: page.truth.length,
    right: split.panels.length === page.truth.length,
    iou: split.iou,
  };
}

/** One arm's totals over a review. */
export interface ArmSummary {
  /** Pages where the arm got the truth's panel count. */
  right: number;
  meanIou: number;
}

/** Totals for arm A and arm B over every page of a review. */
export function reviewSummary(review: SplitReview): [ArmSummary, ArmSummary] {
  const summarize = (arm: 0 | 1): ArmSummary => {
    const outcomes = review.pages.map((page) => armOutcome(page, arm));
    const total = outcomes.reduce((sum, o) => sum + o.iou, 0);
    return {
      right: outcomes.filter((o) => o.right).length,
      meanIou: outcomes.length > 0 ? total / outcomes.length : 0,
    };
  };
  return [summarize(0), summarize(1)];
}

/** The index `delta` steps from `current`, held within a list of `length`. */
export function stepIndex(
  current: number,
  delta: number,
  length: number,
): number {
  if (length === 0) return 0;
  return Math.min(length - 1, Math.max(0, current + delta));
}

/** A page image's URL, resolved against the review.json it is relative to. */
export function pageImageUrl(
  reviewUrl: string,
  image: string,
  base: string,
): string {
  return new URL(image, new URL(reviewUrl, base)).href;
}

/** The selected page's index for a `?page=` value, or the first page. */
export function initialIndex(review: SplitReview, name: string | null): number {
  const index = review.pages.findIndex((page) => page.name === name);
  return index >= 0 ? index : 0;
}
