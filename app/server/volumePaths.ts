/**
 * Where a volume lives under `data/`, and what counts as a safe volume name.
 *
 * A volume is usually one directory (`data/detroit_mich_1929_vol_11`), but a
 * multi-volume atlas is split into subdirectories: `data/brooklyn_1904-1908/vol13`,
 * `data/queens_1950/vol2`. Both forms are first-class -- every API that takes a
 * `volume` accepts either -- so the depth rule and the safety check live here
 * instead of being re-derived per route (issue #228).
 */

/** How many directory levels below `data/` a volume may sit (`data/queens_1950/vol2`). */
export const MAX_VOLUME_DEPTH = 2;

/** Reject a path segment that could escape or re-enter the data directory. */
export function isSafeSegment(name: string): boolean {
  return (
    name !== '' &&
    name !== '.' &&
    name !== '..' &&
    !name.includes('/') &&
    !name.includes('\\')
  );
}

/**
 * Whether ``volume`` is a safe, correctly shaped volume directory name.
 *
 * Accepts a subvolume up to {@link MAX_VOLUME_DEPTH} segments deep. A deeper path
 * is rejected rather than joined, so a volume's own subdirectories (`raw/`,
 * `artifacts/`) can never be addressed as volumes in their own right.
 */
export function isSafeVolume(volume: string): boolean {
  const parts = volume.split('/');
  return parts.length <= MAX_VOLUME_DEPTH && parts.every(isSafeSegment);
}
