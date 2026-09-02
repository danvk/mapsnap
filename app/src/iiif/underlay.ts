/**
 * The key-map underlay (#211): a volume's georeferenced key map drawn beneath
 * its pages, either as the sheet itself or as its P(road) map.
 *
 * Where the street grid has changed since the Sanborn survey there is nothing
 * on OSM for snap to lock onto, and the page lands on an alias. The key map is
 * a Sanborn-era drawing of the same streets, so a page that fails to snap onto
 * the modern grid might snap onto the key map's P(road) map instead. This
 * underlay is the first look at that: does the page's own P(road) map line up
 * with the key map's?
 *
 * The key map's georef is an affine, given by its four corners, so the sheet is
 * placed the way the snap pose overlay is: a maplibre image source at the
 * corner coordinates. The image is fetched through the key map's IIIF image
 * service (an absolute URL from the API, like the pages' tiles) at a bounded
 * size, since a full-resolution sheet is too large for one texture.
 */
import type { KeymapInfo } from '../../server/api';

export type KeymapUnderlayMode = 'off' | 'image' | 'roadprob';

export type Corners = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

/** One key map to draw: an image URL and the corner coordinates to pin it to. */
export interface KeymapUnderlay {
  /** The key map's stem, e.g. "p0"; the layer id derives from it. */
  id: string;
  url: string;
  /** Top-left, top-right, bottom-right, bottom-left, as maplibre wants them. */
  corners: Corners;
}

/** Longest side, in pixels, of the image fetched for the underlay. */
export const UNDERLAY_MAX_PX = 2400;

/** The underlay mode a URL parameter names; anything unknown is off. */
export function underlayFromParam(value: string | null): KeymapUnderlayMode {
  return value === 'image' || value === 'roadprob' ? value : 'off';
}

/** The URL parameter value for a mode; null (remove the parameter) when off. */
export function underlayParam(mode: KeymapUnderlayMode): string | null {
  return mode === 'off' ? null : mode;
}

/** The IIIF request for a service's image, best-fit inside UNDERLAY_MAX_PX square. */
export function underlayImageUrl(
  service: string,
  format: 'jpg' | 'png',
): string {
  return `${service}/full/!${UNDERLAY_MAX_PX},${UNDERLAY_MAX_PX}/0/default.${format}`;
}

/**
 * The underlays to draw in a mode: every key map with a georef and an image
 * service, as its sheet, or in P(road) mode only those with a road-probability
 * map. Nothing when off.
 */
export function keymapUnderlays(
  keymaps: KeymapInfo[],
  mode: KeymapUnderlayMode,
): KeymapUnderlay[] {
  if (mode === 'off') return [];
  const underlays: KeymapUnderlay[] = [];
  for (const keymap of keymaps) {
    if (!keymap.corners || keymap.corners.length !== 4) continue;
    const service =
      mode === 'roadprob' ? keymap.roadprobService : keymap.imageService;
    if (!service) continue;
    underlays.push({
      id: keymap.stem,
      url: underlayImageUrl(service, mode === 'roadprob' ? 'png' : 'jpg'),
      corners: keymap.corners as Corners,
    });
  }
  return underlays;
}
