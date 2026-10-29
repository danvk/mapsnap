/**
 * Volumes on the ground: where each volume of a town lies, and its editions.
 *
 * Zoomed in, the atlas is organized around the volume: one shape per volume
 * of a town (Chicago's vol. 5, or the whole of a small town), whichever
 * editions it has. A click on a shape lists the volume's editions. The shapes
 * come from scripts/atlas/build_footprints.py, one file per town, with an index
 * of each town's bounds so a viewport knows which files it needs.
 */

import { pointInPolygon } from '../geometry';
import { yearsOf, type Volume } from './places';

/** GeoJSON MultiPolygon coordinates: polygons of rings of [lon, lat]. */
export type MultiPolygonCoords = [number, number][][][];

/** [west, south, east, north]. */
export type Bounds = [number, number, number, number];

/** One placed edition's entry in its town's footprint file. */
export interface EditionFootprint {
  year: number | null;
  /** A point inside the footprint, for when there is no clicked point. */
  anchor: [number, number];
  /** Everything the edition's placed sheets cover. */
  footprint: MultiPolygonCoords;
}

/** One volume of a town, across its editions (see group_series). */
export interface SeriesRecord {
  /** The catalog's volume number ("5", "A"); absent when unnumbered. */
  volume?: string;
  /** Every edition, placed or not. */
  items: string[];
  /** A small-scale atlas of the whole city, which the map leaves out. */
  overview?: boolean;
  /** The ground its placed editions cover, or null when none is placed. */
  footprint: MultiPolygonCoords | null;
  anchor: [number, number] | null;
  /**
   * What the map draws: the part no smaller volume covers. `true` when that
   * is the whole footprint; null when smaller volumes cover it all.
   */
  display: true | MultiPolygonCoords | null;
}

/** A town's footprint file. */
export interface TownFootprints {
  volumes: Record<string, EditionFootprint>;
  series: SeriesRecord[];
}

/** The published run the atlas reads annotations from. */
export interface RunInfo {
  /** "v2.0" */
  version: string;
  /** Where the run is published, e.g. https://data.mapsnap.org/runs/v2.0 */
  url: string;
}

/** footprints/index.json. */
export interface FootprintIndex {
  run: RunInfo;
  /** Place id -> bounds around all of the town's footprints. */
  towns: Record<string, Bounds>;
  /**
   * Towns with a footprint file but no shape on the map, because none of
   * their pages is sure enough to draw one (see judge_pages); they keep a dot.
   */
  shapeless?: string[];
}

/**
 * One volume of a town, as the panel shows it: a series from the town's
 * footprint file, or the editions no series holds.
 */
export interface VolumeGroup {
  place: string;
  /** Position in the town's groups (see volumeGroups); a series' index in its file. */
  index: number;
  /** The catalog's volume number, or null. */
  number: string | null;
  overview: boolean;
  /**
   * `series`: a volume the run placed, or numbered. `rest`: the town's
   * editions no series holds, or all of a town with nothing placed.
   */
  kind: 'series' | 'rest';
  /** The editions, newest first. */
  editions: Volume[];
}

/** A volume group of a town. */
export interface GroupRef {
  place: string;
  index: number;
}

/** Whether a point lies inside a MultiPolygon: in an outer ring, and in none of its holes. */
export function inMultiPolygon(
  lng: number,
  lat: number,
  coords: MultiPolygonCoords,
): boolean {
  return coords.some(
    ([outer, ...holes]) =>
      outer !== undefined &&
      pointInPolygon(lng, lat, outer) &&
      !holes.some((hole) => pointInPolygon(lng, lat, hole)),
  );
}

// Shoelace area of a ring, in square degrees -- only ever compared.
function ringArea(ring: [number, number][]): number {
  let twice = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i] as [number, number];
    const [xj, yj] = ring[j] as [number, number];
    twice += xj * yi - xi * yj;
  }
  return Math.abs(twice) / 2;
}

/** A MultiPolygon's area in square degrees (outer rings less holes), for ranking. */
export function multiPolygonArea(coords: MultiPolygonCoords): number {
  return coords.reduce(
    (sum, [outer, ...holes]) =>
      sum +
      (outer ? ringArea(outer) : 0) -
      holes.reduce((h, hole) => h + ringArea(hole), 0),
    0,
  );
}

/** [west, south, east, north] around a MultiPolygon. */
export function multiPolygonBounds(coords: MultiPolygonCoords): Bounds {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const polygon of coords) {
    for (const [lng, lat] of polygon[0] ?? []) {
      west = Math.min(west, lng);
      east = Math.max(east, lng);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
    }
  }
  return [west, south, east, north];
}

/** The towns whose footprints reach into a [west, south, east, north] view. */
export function townsInView(
  towns: Record<string, Bounds>,
  view: Bounds,
): string[] {
  const [west, south, east, north] = view;
  return Object.entries(towns)
    .filter(
      ([, [w, s, e, n]]) => w <= east && e >= west && s <= north && n >= south,
    )
    .map(([place]) => place);
}

/** Newest first, undated last; ties in catalog order. */
function newestFirst(a: Volume, b: Volume): number {
  if (a.year === b.year) return 0;
  if (a.year === null) return 1;
  if (b.year === null) return -1;
  return b.year - a.year;
}

/**
 * A town's volumes: each series in its footprint file, then the editions none
 * of them holds.
 *
 * A town with no footprint file -- the run placed none of it -- is one group
 * of every edition. So is the remainder of a town that has series: the
 * unnumbered editions of a numbered town that nothing places, and those on
 * paper only.
 */
export function volumeGroups(
  place: string,
  townVolumes: Volume[],
  town: TownFootprints | null,
): VolumeGroup[] {
  const byItem = new Map(townVolumes.map((volume) => [volume.item, volume]));
  const groups: VolumeGroup[] = (town?.series ?? []).map((series, index) => ({
    place,
    index,
    number: series.volume ?? null,
    overview: series.overview ?? false,
    kind: 'series',
    editions: series.items
      .map((item) => byItem.get(item))
      .filter((volume): volume is Volume => volume !== undefined)
      .sort(newestFirst),
  }));
  const held = new Set(town?.series.flatMap((series) => series.items) ?? []);
  const rest = townVolumes.filter((volume) => !held.has(volume.item));
  if (rest.length > 0) {
    groups.push({
      place,
      index: groups.length,
      number: null,
      overview: false,
      kind: 'rest',
      editions: [...rest].sort(newestFirst),
    });
  }
  return groups;
}

/** The group holding an edition, if any. */
export function groupOfItem(
  groups: VolumeGroup[],
  item: string,
): VolumeGroup | null {
  return (
    groups.find((group) => group.editions.some((v) => v.item === item)) ?? null
  );
}

/** "1894–1950", "1950" or "undated": the years a group's editions span. */
export function yearSpan(editions: Volume[]): string {
  const years = yearsOf(editions).filter(
    (year): year is number => year !== null,
  );
  if (years.length === 0) return 'undated';
  const first = years[years.length - 1];
  const last = years[0];
  return first === last ? String(first) : `${first}–${last}`;
}

/**
 * A group's name in the panel: "Vol. 5", or the town's own name for an
 * unnumbered volume, with its years when the town has more than one.
 */
export function groupLabel(
  group: VolumeGroup,
  townName: string,
  groupCount: number,
): string {
  if (group.overview) return `Overview atlas, ${yearSpan(group.editions)}`;
  if (group.number) return `Vol. ${group.number}`;
  if (group.kind === 'rest' && groupCount > 1) return 'Other volumes';
  return groupCount > 1 ? `${townName}, ${yearSpan(group.editions)}` : townName;
}

/**
 * The volumes whose footprint holds a point, across the loaded towns, most
 * specific first.
 *
 * Smaller footprints first, since the smaller map of a spot is the more
 * detailed one. The first is what a click on the map selects; the rest are
 * listed as also covering the spot. Overview atlases are never found here:
 * they cover everything, so they are reached only from the town's list.
 */
export function seriesAt(
  lng: number,
  lat: number,
  towns: ReadonlyMap<string, TownFootprints>,
): GroupRef[] {
  const hits: { ref: GroupRef; area: number }[] = [];
  for (const [place, town] of towns) {
    town.series.forEach((series, index) => {
      if (
        series.overview ||
        !series.footprint ||
        !inMultiPolygon(lng, lat, series.footprint)
      ) {
        return;
      }
      hits.push({
        ref: { place, index },
        area: multiPolygonArea(series.footprint),
      });
    });
  }
  hits.sort((a, b) => a.area - b.area);
  return hits.map((hit) => hit.ref);
}

/**
 * The edition to open a volume on.
 *
 * The newest placed edition covering the clicked point, else the newest placed
 * edition, else the newest digitized one, else the newest of any kind.
 */
export function defaultEdition(
  editions: Volume[],
  footprints: Record<string, EditionFootprint>,
  point: [number, number] | null,
): Volume | null {
  const placed = editions.filter((volume) => footprints[volume.item]);
  const here = point
    ? placed.find((volume) =>
        inMultiPolygon(point[0], point[1], footprints[volume.item]!.footprint),
      )
    : undefined;
  return (
    here ??
    placed[0] ??
    editions.find((volume) => volume.state) ??
    editions[0] ??
    null
  );
}

/** One of the edition buttons of a volume. */
export interface EditionOption {
  volume: Volume;
  /** "1950", or "1950 (2)" for the second edition of a year. */
  label: string;
  /**
   * `placed`: the run placed some of it. `unplaced`: digitized, nothing placed.
   * `not digitized`: on paper only at the Library of Congress.
   */
  status: 'placed' | 'unplaced' | 'not digitized';
}

/** A volume's editions as buttons, newest first. */
export function editionOptions(
  editions: Volume[],
  footprints: Record<string, EditionFootprint>,
): EditionOption[] {
  const ofYear = new Map<number | null, number>();
  for (const volume of editions) {
    ofYear.set(volume.year, (ofYear.get(volume.year) ?? 0) + 1);
  }
  const seen = new Map<number | null, number>();
  return editions.map((volume) => {
    const nth = (seen.get(volume.year) ?? 0) + 1;
    seen.set(volume.year, nth);
    const year = volume.year === null ? 'undated' : String(volume.year);
    return {
      volume,
      label: (ofYear.get(volume.year) ?? 0) > 1 ? `${year} (${nth})` : year,
      status: footprints[volume.item]
        ? 'placed'
        : volume.state
          ? 'unplaced'
          : 'not digitized',
    };
  });
}

/** A footprint feature's id: its town and series. */
export function groupKey(ref: GroupRef): string {
  return `${ref.place}#${ref.index}`;
}

/**
 * The map's volume layer: each loaded volume's displayed shape.
 *
 * The selected volume is drawn whole, whatever part of it smaller volumes
 * cover, so its outline frames the imagery on screen.
 */
export function footprintFeatures(
  towns: ReadonlyMap<string, TownFootprints>,
  selected: GroupRef | null,
): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const [place, town] of towns) {
    town.series.forEach((series, index) => {
      const isSelected =
        selected !== null &&
        selected.place === place &&
        selected.index === index;
      const shape = isSelected
        ? series.footprint
        : series.display === true
          ? series.footprint
          : series.display;
      if (!shape) return;
      const key = groupKey({ place, index });
      features.push({
        type: 'Feature',
        id: key,
        geometry: { type: 'MultiPolygon', coordinates: shape },
        properties: { key, place, index, selected: isSelected ? 1 : 0 },
      });
    });
  }
  return { type: 'FeatureCollection', features };
}
