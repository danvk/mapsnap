/**
 * The selection after clicking shapes `hit`: those shapes, or nothing when
 * they are already exactly what is selected.
 *
 * Panels tile the whole sheet, so every click lands in one; clicking the
 * selected panel again is how a panel gets deselected.
 */
export function toggledSelection(
  current: ReadonlySet<number>,
  hit: number[],
): Set<number> {
  const next = new Set(hit);
  const same =
    next.size > 0 &&
    next.size === current.size &&
    [...next].every((i) => current.has(i));
  return same ? new Set() : next;
}
