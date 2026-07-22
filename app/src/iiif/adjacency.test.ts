import { describe, expect, it } from 'vitest';
import { adjacencyClaimFeatures, mutualNumbersByStem } from './adjacency';
import type { AdjacencyData, AdjacencyDetection } from '../types';
import type { PageGeo } from './pages';

// A 1000×1000 page on a 0.01°×0.01° geo box at (-74, 40.7).
function page(stem: string): PageGeo {
  return {
    stem,
    itemIndex: 0,
    pageKey: stem,
    splitIndex: null,
    width: 1000,
    height: 1000,
    corners: [
      [-74, 40.7],
      [-73.99, 40.7],
      [-73.99, 40.69],
      [-74, 40.69],
    ],
    rectRing: [],
    clipRing: [],
    scalePixelsPerFoot: 1,
    rotationDegrees: 0,
    gcps: [],
    transformationType: 'polynomial',
  };
}

function detection(
  number: number,
  polygon: [number, number][],
  claim: boolean,
): AdjacencyDetection {
  return {
    number,
    text: String(number),
    confidence: 1,
    polygon,
    height: 30,
    x_frac: 0,
    y_frac: 0,
    edge: 'center',
    claim,
  };
}

describe('mutualNumbersByStem', () => {
  it("collects each page's reciprocated-neighbour numbers", () => {
    const adjacency: AdjacencyData = {
      pages: {
        p1: { number: 1, detections: [] },
        p2: { number: 2, detections: [] },
        p3: { number: 3, detections: [] },
      },
      adjacency: [['p1', 'p2']],
    };
    const mutual = mutualNumbersByStem(adjacency);
    expect([...(mutual.get('p1') ?? [])]).toEqual([2]);
    expect([...(mutual.get('p2') ?? [])]).toEqual([1]);
    expect(mutual.has('p3')).toBe(false); // no reciprocated edge
  });
});

describe('adjacencyClaimFeatures', () => {
  const adjacency: AdjacencyData = {
    pages: {
      p1: {
        number: 1,
        width: 1000,
        height: 1000,
        detections: [
          detection(
            2,
            [
              [0, 0],
              [100, 0],
              [100, 100],
              [0, 100],
            ],
            true,
          ), // claims p2 (mutual)
          detection(
            3,
            [
              [500, 500],
              [600, 500],
              [600, 600],
              [500, 600],
            ],
            true,
          ), // claims p3 (one-sided)
          detection(9, [[0, 0]], false), // not a claim
        ],
      },
      p2: { number: 2, detections: [] },
      p3: { number: 3, detections: [] },
    },
    adjacency: [['p1', 'p2']],
  };

  it('returns a closed geo ring per claim, flagging mutual vs one-sided', () => {
    const claims = adjacencyClaimFeatures(adjacency, [page('p1')]);
    expect(claims).toHaveLength(2); // the two claims; the non-claim detection is skipped
    expect(claims.map((c) => c.mutual)).toEqual([true, false]);
    expect(claims.map((c) => c.stem)).toEqual(['p1', 'p1']); // both drawn on p1
    // First box's top-left pixel (0,0) maps to the page's NW corner; the ring is closed.
    expect(claims[0]!.ring[0]).toEqual([-74, 40.7]);
    expect(claims[0]!.ring).toHaveLength(5);
    expect(claims[0]!.ring[4]).toEqual(claims[0]!.ring[0]);
  });

  it('skips pages absent from the adjacency data', () => {
    expect(adjacencyClaimFeatures(adjacency, [page('p99')])).toEqual([]);
  });
});
