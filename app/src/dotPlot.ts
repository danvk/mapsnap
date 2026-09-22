/**
 * Laying out a flat-bottom dot plot: one dot per value, stacked, never binned.
 *
 * Every point is drawn at its own x, so the plot cannot invent or hide a mode
 * the way a histogram's bin edges can. Dots that would overlap stack upwards
 * from the baseline, which is what gives the pile its shape -- a Wilkinson dot
 * plot rather than the sinusoidal spread of a beeswarm.
 *
 * Its own module because the packing is worth testing directly, and because
 * importing the component to reach it pulls in React.
 */

/** Where one dot goes, in SVG coordinates, and which row of the pile it is in. */
export interface DotPosition {
  x: number;
  y: number;
  row: number;
}

export interface DotLayout {
  dots: DotPosition[];
  /** Deepest pile, before any squashing. */
  rows: number;
  /** Value at x=0 and at x=width, after the degenerate-range widening below. */
  domain: [number, number];
  /** Vertical step actually used; less than a diameter once a pile is squashed. */
  step: number;
}

/** Maps between a metric's values and x pixels, both ways. */
export interface ValueScale {
  toX: (value: number) => number;
  toValue: (x: number) => number;
}

/**
 * The scale for a domain, inset by the dot radius at each end.
 *
 * Exported so the chart converts a drag back to values with exactly the
 * arithmetic that placed the dots; two copies of it would drift by a pixel and
 * brush the wrong point.
 */
export function scaleFor(
  domain: [number, number],
  width: number,
  radius: number,
): ValueScale {
  const [low, high] = domain;
  const span = high - low || 1;
  const usable = Math.max(1, width - 2 * radius);
  return {
    toX: (value) => radius + ((value - low) / span) * usable,
    toValue: (x) => low + ((x - radius) / usable) * span,
  };
}

/** The domain to plot: the data's range, widened when every value is identical. */
export function domainOf(values: number[]): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (high > low) return [low, high];
  // Every value identical: an arbitrary unit of padding, so the pile sits in
  // the middle rather than at x=0 and the axis labels read sensibly.
  const pad = Math.abs(low) > 0 ? Math.abs(low) * 0.05 : 0.5;
  return [low - pad, high + pad];
}

/**
 * A fixed permutation of 0..n-1: the order dots are placed in.
 *
 * Placing in ascending x gives every dense cluster a visible up-and-right
 * diagonal -- consecutive values fill rows 0, 1, 2 as x creeps rightward, so
 * the pile leans. The slope is an artifact of the traversal and says nothing
 * about the data. Shuffling removes it.
 *
 * Derived from the index alone so it is stable: a re-render that reshuffled
 * would make every dot jump.
 */
export function stackOrder(count: number): number[] {
  const key = (index: number) => {
    let hash = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
    hash ^= hash >>> 13;
    return Math.imul(hash, 0xc2b2ae35) >>> 0;
  };
  return Array.from({ length: count }, (_unused, index) => index).sort(
    (a, b) => key(a) - key(b),
  );
}

/**
 * Positions for every value, in the order given.
 *
 * A dot goes in the lowest row where it clears every dot already there by a
 * diameter, so piles grow upward from the baseline. Dots are placed in a
 * shuffled order rather than left to right, which is what keeps a dense cluster
 * from leaning (see stackOrder); that in turn means a row's occupants are not
 * sorted, so each candidate row is checked against all of them rather than just
 * its rightmost.
 *
 * The vertical step is then whatever makes the deepest pile fit the height: a
 * diameter when there is room, less when there is not. Squashing rather than
 * dropping dots or growing the chart is deliberate -- these distributions are
 * often near-degenerate (a volume whose every page shares one rotation piles
 * 90-odd dots into a single column), and a solid bar is the honest picture of
 * that.
 */
export function layoutDots(
  values: number[],
  width: number,
  height: number,
  radius: number,
): DotLayout {
  if (values.length === 0) {
    return { dots: [], rows: 0, domain: [0, 1], step: 2 * radius };
  }
  const domain = domainOf(values);
  const { toX } = scaleFor(domain, width, radius);
  const xs = values.map(toX);

  const occupied: number[][] = [];
  const rows = new Array<number>(values.length);
  for (const index of stackOrder(values.length)) {
    const x = xs[index] ?? 0;
    let row = occupied.findIndex((placed) =>
      placed.every((other) => Math.abs(x - other) >= 2 * radius),
    );
    if (row === -1) {
      row = occupied.length;
      occupied.push([x]);
    } else {
      occupied[row]?.push(x);
    }
    rows[index] = row;
  }

  const deepest = Math.max(...rows) + 1;
  const step =
    deepest > 1
      ? Math.min(2 * radius, (height - 2 * radius) / (deepest - 1))
      : 2 * radius;
  const dots = values.map((_value, index) => {
    const row = rows[index] ?? 0;
    return { x: xs[index] ?? 0, y: height - radius - row * step, row };
  });
  return { dots, rows: deepest, domain, step };
}

/** The dot nearest a pointer, or null when the pointer is not on one. */
export function dotAt(
  layout: DotLayout,
  x: number,
  y: number,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestDistance = Infinity;
  layout.dots.forEach((dot, index) => {
    const distance = Math.hypot(dot.x - x, dot.y - y);
    // A squashed pile puts dots closer together than their radius, so allow a
    // little slack rather than requiring a hit on the drawn circle.
    if (distance <= radius + 2 && distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}
