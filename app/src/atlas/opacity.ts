/** The opacities the `p` key steps through, in order. */
const OPACITY_STEPS = [100, 50, 0];

/**
 * The sheet opacity after one press of `p`: 100 -> 50 -> 0 -> 100.
 *
 * From a value the slider left between steps, the cycle starts over at 100.
 */
export function nextOpacity(current: number): number {
  const index = OPACITY_STEPS.indexOf(current);
  return OPACITY_STEPS[(index + 1) % OPACITY_STEPS.length] ?? 100;
}
