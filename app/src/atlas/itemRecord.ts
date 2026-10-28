/**
 * A volume's item record: every sheet, and what the run did with each one.
 *
 * Built by scripts/atlas/build_item_pages.py and published beside the runs at
 * data.mapsnap.org/items/<item>.json; mapsnap.org/item draws its page from
 * the same file. The atlas reads it for what an annotation cannot say: the
 * images a run did not place, and which sheet (`?sp=`) an image is.
 */

import { panelCrop, panelIndexFromStem, parentStem } from '../panelCrop';
import type { PanelPolygon } from '../types';

export const DATA_URL = 'https://data.mapsnap.org';
export const ITEM_PAGE = 'https://mapsnap.org/item';

/** What a run did with one image: a whole sheet, or one panel of a split one. */
export interface ItemFit {
  /** Page key, with a panel suffix for a split sheet: "p9__2". */
  image: string;
  status: 'placed' | 'key map' | 'unplaced' | 'withheld';
  /** e.g. "tier 1, 31.56559,-85.24542 +/-103m" for an unplaced image. */
  note?: string;
}

/** One mirrored sheet of a volume. */
export interface ItemSheet {
  /** Its position at loc.gov, `?sp=` on the item page. */
  seq: number;
  /** Page key, e.g. "p2". */
  key: string;
  /** The Chronoscope image's size: a 25% copy of LoC's scan. */
  width: number;
  height: number;
  /** The Chronoscope image service. */
  image: string;
  thumbnail: { url: string; width: number; height: number };
  loc_page?: string;
  fit?: ItemFit[];
  /**
   * A split sheet's panels in panel order, their outlines in the Chronoscope
   * image's pixels, and where the item page puts each one's number.
   */
  panels?: { points: PanelPolygon; label?: [number, number] }[];
}

export interface ItemRecord {
  id: string;
  name: string;
  date: string;
  sheets: ItemSheet[];
}

/** An image the run did not place, cut out of its sheet when it is a panel. */
export interface UnplacedImage {
  /** "p2__3", or "p2" for a whole sheet. */
  image: string;
  sheet: ItemSheet;
  status: 'unplaced' | 'withheld';
  note: string | null;
  /** The panel's box and outline in the sheet's pixels; the whole sheet when null. */
  crop: {
    x: number;
    y: number;
    width: number;
    height: number;
    ring: PanelPolygon;
  } | null;
}

/** Fetch a volume's item record; null when there is none. */
export async function fetchItemRecord(
  item: string,
  dataUrl: string = DATA_URL,
): Promise<ItemRecord | null> {
  try {
    const response = await fetch(`${dataUrl}/items/${item}.json`);
    return response.ok ? ((await response.json()) as ItemRecord) : null;
  } catch {
    return null;
  }
}

/**
 * The images of a volume the run did not place, in sheet order.
 *
 * Withheld images were placed and then dropped when the run was published
 * (an implausibly large sheet, too far from town), so to the map they are
 * unplaced too. Key maps are left out: the run places them in a file of their
 * own.
 */
export function unplacedImages(record: ItemRecord): UnplacedImage[] {
  const images: UnplacedImage[] = [];
  for (const sheet of record.sheets) {
    for (const fit of sheet.fit ?? []) {
      if (fit.status !== 'unplaced' && fit.status !== 'withheld') continue;
      const panel = panelIndexFromStem(fit.image);
      const crop =
        panel !== null && sheet.panels
          ? panelCrop(
              sheet.panels.map((entry) => entry.points),
              panel,
              sheet.width,
              sheet.height,
            )
          : null;
      images.push({
        image: fit.image,
        sheet,
        status: fit.status,
        note: fit.note ?? null,
        crop,
      });
    }
  }
  return images;
}

/** The sheet a page stem ("p2__1", "p100W") is on, if the record has it. */
export function sheetOfStem(
  record: ItemRecord,
  stem: string,
): ItemSheet | null {
  const key = parentStem(stem);
  return record.sheets.find((sheet) => sheet.key === key) ?? null;
}

/** mapsnap.org's page for a volume, optionally open on one sheet. */
export function itemViewerUrl(item: string, seq?: number): string {
  const query = new URLSearchParams({ id: item });
  if (seq !== undefined) query.set('sp', String(seq));
  return `${ITEM_PAGE}?${query}`;
}

/** Which of a Chronoscope image's sizes: the full 25% copy, or a half or quarter of it. */
export type ImageScale = 1 | 2 | 4;

/**
 * A size of a Chronoscope image, as a level-0 IIIF URL.
 *
 * The CDN serves only the sizes its info.json lists: the full image (as
 * `max`; its explicit size is a 404), then halving, each rounded up.
 */
export function chronoscopeImageUrl(
  sheet: ItemSheet,
  scale: ImageScale,
): string {
  if (scale === 1) return `${sheet.image}/full/max/0/default.jpg`;
  const width = Math.ceil(sheet.width / scale);
  const height = Math.ceil(sheet.height / scale);
  return `${sheet.image}/full/${width},${height}/0/default.jpg`;
}

/** The smallest of a sheet's sizes that is at least `width` pixels across. */
export function scaleForWidth(sheetWidth: number, width: number): ImageScale {
  if (Math.ceil(sheetWidth / 4) >= width) return 4;
  if (Math.ceil(sheetWidth / 2) >= width) return 2;
  return 1;
}
