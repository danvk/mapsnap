/**
 * Adjacency-claim geometry for the volume viewer's overlay.
 *
 * `mapsnap adjacency` records, per page, the sheet numbers it read near its edges (a "claim"
 * points at a neighbouring page) plus the graph of reciprocated (mutual) claims. This maps each
 * claim's pixel box into geo through its page's georeference so the viewer can draw it where it
 * sits on the map, blue when the claimed neighbour claims back and amber when the claim is
 * one-sided.
 */

import { projectThroughCorners } from '../geometry';
import type { AdjacencyData } from '../types';
import type { PageGeo } from './pages';

/** A claim box to draw on the map: a closed [lon, lat] ring, mutual when its neighbour reciprocates. */
export interface AdjacencyClaim {
  ring: [number, number][];
  /** True when the claimed page number belongs to a reciprocated (mutual) neighbour. */
  mutual: boolean;
  /** File stem of the page the claim is drawn on, so the viewer can dim other pages' claims. */
  stem: string;
}

/**
 * Each page's reciprocated-neighbour page numbers, keyed by page stem.
 *
 * The adjacency graph's edges are exactly the mutual pairs, so a page's claim is mutual when the
 * number it claims is one of its neighbours' numbers here.
 */
export function mutualNumbersByStem(
  adjacency: AdjacencyData,
): Map<string, Set<number>> {
  const result = new Map<string, Set<number>>();
  const add = (stem: string, number: number | null | undefined): void => {
    if (number == null) return;
    const set = result.get(stem) ?? new Set<number>();
    set.add(number);
    result.set(stem, set);
  };
  for (const [a, b] of adjacency.adjacency) {
    add(a, adjacency.pages[b]?.number);
    add(b, adjacency.pages[a]?.number);
  }
  return result;
}

/**
 * Claim boxes for the given pages, each polygon mapped from its page's pixel frame into geo
 * through the page's georeference corners. Only `claim` detections are returned; a page absent
 * from the adjacency data contributes none.
 */
export function adjacencyClaimFeatures(
  adjacency: AdjacencyData,
  pages: PageGeo[],
): AdjacencyClaim[] {
  const mutualByStem = mutualNumbersByStem(adjacency);
  const claims: AdjacencyClaim[] = [];
  for (const page of pages) {
    const adjacencyPage = adjacency.pages[page.stem];
    if (!adjacencyPage) continue;
    const width = adjacencyPage.width ?? page.width;
    const height = adjacencyPage.height ?? page.height;
    const mutualNumbers = mutualByStem.get(page.stem) ?? new Set<number>();
    for (const detection of adjacencyPage.detections) {
      if (!detection.claim || detection.polygon.length === 0) continue;
      const ring = detection.polygon.map(([x, y]) =>
        projectThroughCorners(page.corners, width, height, x, y),
      );
      ring.push(ring[0]!); // close the polygon
      claims.push({
        ring,
        mutual: mutualNumbers.has(detection.number),
        stem: page.stem,
      });
    }
  }
  return claims;
}
