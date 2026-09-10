/**
 * Which raster the volume viewer draws for each page (#352): the sheet
 * itself, its P(region) map (`mapsnap region`), or its P(road) map.
 *
 * The maps are served through the same annotation as the sheets, each page's
 * image service pointed at the PNG instead, so they warp and clip exactly as
 * the sheets do: where two neighbours' content regions overlap on the map, the
 * pages overlap on the ground.
 */
import type { PageImage } from '../../server/iiifAnnotations';

export type { PageImage };

/** The image a URL parameter names; anything unknown is the sheet. */
export function pageImageFromParam(value: string | null): PageImage {
  return value === 'region' || value === 'roadprob' ? value : 'page';
}

/** The URL parameter value for an image; null (omit) for the default sheet. */
export function pageImageParam(image: PageImage): string | null {
  return image === 'page' ? null : image;
}

/** What the status line calls a page's alternate image when it is missing. */
export function pageImageNoun(image: PageImage): string {
  return image === 'region' ? 'region map' : 'P(road) map';
}
