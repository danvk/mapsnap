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
 * Each key map is drawn from a georeference annotation the server builds
 * (`/iiif-api/keymap-annotation`), warped by a thin-plate spline through the
 * sheet's own GCPs -- the model keymap-snap places pages in, whose residuals
 * are about half the shipped affine's. Rendering it with allmaps, the way the
 * pages are rendered, is what makes the spline visible: a four-corner image
 * would show the affine placement the pipeline does not use.
 */
import type { KeymapInfo } from '../../server/api';

/** Which image to draw for a key map: the sheet, or its road-probability map. */
export type KeymapUnderlayImage = 'sheet' | 'roadprob';

/** One key map to draw, as the annotation URL that places it. */
export interface KeymapUnderlay {
  /** The key map's stem, e.g. "p0". */
  id: string;
  /** Georeference annotation URL; allmaps fetches and warps it. */
  annotationUrl: string;
}

/** The image a URL parameter names; anything unknown is the sheet. */
export function underlayImageFromParam(
  value: string | null,
): KeymapUnderlayImage {
  return value === 'roadprob' ? 'roadprob' : 'sheet';
}

/** The URL parameter value for an image; null (omit) for the default sheet. */
export function underlayImageParam(image: KeymapUnderlayImage): string | null {
  return image === 'roadprob' ? 'roadprob' : null;
}

/**
 * The underlays to draw for a volume: every key map that has a georeference,
 * or in P(road) mode only those that also have a road-probability map.
 */
export function keymapUnderlays(
  volume: string,
  keymaps: KeymapInfo[],
  image: KeymapUnderlayImage,
): KeymapUnderlay[] {
  const underlays: KeymapUnderlay[] = [];
  for (const keymap of keymaps) {
    if (!keymap.hasGeoref) continue;
    if (image === 'roadprob' && !keymap.hasRoadprob) continue;
    const query = new URLSearchParams({
      volume,
      stem: keymap.stem,
      image,
    });
    underlays.push({
      id: keymap.stem,
      annotationUrl: `/iiif-api/keymap-annotation?${query}`,
    });
  }
  return underlays;
}
