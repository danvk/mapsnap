/**
 * Volume footprints: where each digitized volume's sheets lie on the ground.
 *
 * Zoomed in, the atlas is organised around the volume rather than the town:
 * the map shows each town's newest coverage, a click picks the newest volume
 * covering that spot, and the panel offers the same spot in other years. The
 * footprints come from scripts/atlas/build_footprints.py, one file per town,
 * with an index of each town's bounds so a viewport knows which files it needs.
 */

import { pointInPolygon } from '../geometry';
import { volumesOfYear, yearsOf, type Volume } from './places';

/** GeoJSON MultiPolygon coordinates: polygons of rings of [lon, lat]. */
export type MultiPolygonCoords = [number, number][][][];

/** One volume's entry in its town's footprint file. */
export interface VolumeFootprint {
  year: number | null;
  /** A point inside the footprint, for when there is no clicked point. */
  anchor: [number, number];
  /** Everything the volume's placed sheets cover. */
  footprint: MultiPolygonCoords;
  /**
   * What the map draws: the part no later volume of the town covers. `true`
   * when that is the whole footprint; null when later volumes cover it all.
   */
  display: true | MultiPolygonCoords | null;
}

/** A town's footprint file: item -> footprint. */
export type TownFootprints = Record<string, VolumeFootprint>;

/** place id -> [west, south, east, north] around all of the town's footprints. */
export type FootprintIndex = Record<string, [number, number, number, number]>;

/** A volume of a town. */
export interface VolumeRef {
  item: string;
  place: string;
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
export function multiPolygonBounds(
  coords: MultiPolygonCoords,
): [number, number, number, number] {
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
  index: FootprintIndex,
  view: [number, number, number, number],
): string[] {
  const [west, south, east, north] = view;
  return Object.entries(index)
    .filter(
      ([, [w, s, e, n]]) => w <= east && e >= west && s <= north && n >= south,
    )
    .map(([place]) => place);
}

/**
 * The newest volume whose footprint contains a point, across the loaded towns.
 *
 * Newest by year, undated last; between two of one year the smaller footprint
 * wins, being the more specific map of that spot.
 */
export function newestVolumeAt(
  lng: number,
  lat: number,
  towns: ReadonlyMap<string, TownFootprints>,
): VolumeRef | null {
  let best: { ref: VolumeRef; year: number; area: number } | null = null;
  for (const [place, volumes] of towns) {
    for (const [item, volume] of Object.entries(volumes)) {
      if (!inMultiPolygon(lng, lat, volume.footprint)) continue;
      const year = volume.year ?? -Infinity;
      const area = multiPolygonArea(volume.footprint);
      if (
        !best ||
        year > best.year ||
        (year === best.year && area < best.area)
      ) {
        best = { ref: { item, place }, year, area };
      }
    }
  }
  return best?.ref ?? null;
}

/** One of the year buttons for a selected volume. */
export interface YearOption {
  year: number | null;
  /** The volume to open, or null when there is nothing to open. */
  item: string | null;
  /**
   * `placed`: a digitized volume whose footprint covers the spot. `unplaced`:
   * a digitized volume with no sheet placed, so nothing is known of where it
   * is. `not digitized`: on paper only at the Library of Congress.
   */
  status: 'placed' | 'unplaced' | 'not digitized';
  /** How many volumes of the year the option stands for (not digitized). */
  count: number;
}

/**
 * The years a spot of a town can be seen in, newest first.
 *
 * Only the selected volume's own town is searched, and a year is offered when
 * one of its volumes covers the spot -- the smallest, when several do. A
 * year whose volumes have no footprint (on paper only, or digitized but never
 * placed) is offered too, since nothing rules out their covering the spot; a
 * year whose every volume covers somewhere else is left out.
 */
export function yearOptions(
  point: [number, number],
  selected: string,
  townVolumes: Volume[],
  footprints: TownFootprints,
): YearOption[] {
  const [lng, lat] = point;
  const options: YearOption[] = [];
  for (const year of yearsOf(townVolumes)) {
    const ofYear = volumesOfYear(townVolumes, year);
    if (ofYear.some((volume) => volume.item === selected)) {
      options.push({ year, item: selected, status: 'placed', count: 1 });
      continue;
    }
    const covering = ofYear
      .filter((volume) => {
        const footprint = footprints[volume.item];
        return footprint && inMultiPolygon(lng, lat, footprint.footprint);
      })
      .sort(
        (a, b) =>
          multiPolygonArea(footprints[a.item]!.footprint) -
          multiPolygonArea(footprints[b.item]!.footprint),
      );
    if (covering[0]) {
      options.push({
        year,
        item: covering[0].item,
        status: 'placed',
        count: 1,
      });
      continue;
    }
    const unknown = ofYear.filter((volume) => !footprints[volume.item]);
    const digitized = unknown.find((volume) => volume.state);
    if (digitized) {
      options.push({
        year,
        item: digitized.item,
        status: 'unplaced',
        count: 1,
      });
    } else if (unknown.length > 0) {
      options.push({
        year,
        item: null,
        status: 'not digitized',
        count: unknown.length,
      });
    }
  }
  return options;
}

/**
 * The map's footprint layer: each loaded volume's displayed shape.
 *
 * The selected volume is drawn whole, whatever part of it later volumes
 * cover, so its outline frames the imagery on screen.
 */
export function footprintFeatures(
  towns: ReadonlyMap<string, TownFootprints>,
  selected: string | null,
): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const [place, volumes] of towns) {
    for (const [item, volume] of Object.entries(volumes)) {
      const isSelected = item === selected;
      const shape = isSelected
        ? volume.footprint
        : volume.display === true
          ? volume.footprint
          : volume.display;
      if (!shape) continue;
      features.push({
        type: 'Feature',
        id: item,
        geometry: { type: 'MultiPolygon', coordinates: shape },
        properties: {
          item,
          place,
          year: volume.year,
          selected: isSelected ? 1 : 0,
        },
      });
    }
  }
  return { type: 'FeatureCollection', features };
}
