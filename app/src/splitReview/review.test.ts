import { describe, expect, it } from 'vitest';
import {
  armOutcome,
  initialIndex,
  pageImageUrl,
  reviewSummary,
  stepIndex,
} from './review';
import type { ReviewPage, SplitReview } from './types';

const square = (x: number): [number, number][] => [
  [x, 0],
  [x + 10, 0],
  [x + 10, 10],
  [x, 10],
];

function page(name: string, ious: [number, number], counts: [number, number]) {
  const arm = (n: number, iou: number) => ({
    panels: Array.from({ length: n }, (_, i) => square(i * 10)),
    iou,
  });
  return {
    name,
    image: `images/${name}.jpg`,
    title: '',
    item: 'item',
    page: 'p1',
    label: 'split',
    width: 20,
    height: 10,
    truth: [square(0), square(10)],
    arms: [arm(counts[0], ious[0]), arm(counts[1], ious[1])],
  } satisfies ReviewPage;
}

const review: SplitReview = {
  title: 'test',
  arms: ['classical', 'model'],
  pages: [page('a', [0.5, 1], [1, 2]), page('b', [0.9, 0.7], [2, 3])],
};

describe('armOutcome', () => {
  it('compares the panel count with the truth', () => {
    expect(armOutcome(review.pages[0]!, 0)).toEqual({
      panels: 1,
      truthPanels: 2,
      right: false,
      iou: 0.5,
    });
    expect(armOutcome(review.pages[0]!, 1).right).toBe(true);
  });
});

describe('reviewSummary', () => {
  it('totals right counts and mean IoU per arm', () => {
    const [a, b] = reviewSummary(review);
    expect(a.right).toBe(1);
    expect(a.meanIou).toBeCloseTo(0.7);
    expect(b.right).toBe(1);
    expect(b.meanIou).toBeCloseTo(0.85);
  });

  it('is zero for an empty review', () => {
    expect(reviewSummary({ ...review, pages: [] })[0]).toEqual({
      right: 0,
      meanIou: 0,
    });
  });
});

describe('stepIndex', () => {
  it('steps and stops at either end', () => {
    expect(stepIndex(0, 1, 3)).toBe(1);
    expect(stepIndex(2, 1, 3)).toBe(2);
    expect(stepIndex(0, -1, 3)).toBe(0);
    expect(stepIndex(0, 1, 0)).toBe(0);
  });
});

describe('pageImageUrl', () => {
  it('resolves an image beside the review.json', () => {
    expect(
      pageImageUrl(
        'data/split-review/test-100/review.json',
        'images/a.jpg',
        'http://localhost:5173/mapsnap/split-review.html?review=x',
      ),
    ).toBe(
      'http://localhost:5173/mapsnap/data/split-review/test-100/images/a.jpg',
    );
  });
});

describe('initialIndex', () => {
  it('finds a named page, else the first', () => {
    expect(initialIndex(review, 'b')).toBe(1);
    expect(initialIndex(review, 'zzz')).toBe(0);
    expect(initialIndex(review, null)).toBe(0);
  });
});
