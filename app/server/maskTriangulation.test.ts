import { describe, expect, it } from 'vitest';
import { failureRates, triangulationFailures } from './maskTriangulation.ts';

type Point = [number, number];

/** A one-map georeference annotation. */
function annotation(
  mask: Point[],
  gcps: Point[],
  [width, height]: Point = [4000, 4000],
) {
  const points = mask.map(([x, y]) => `${x},${y}`).join(' ');
  return {
    '@context': 'http://iiif.io/api/extension/georef/1/context.json',
    type: 'Annotation',
    motivation: 'georeferencing',
    target: {
      type: 'SpecificResource',
      source: {
        id: 'https://example.org/iiif/sheet',
        type: 'ImageService2',
        width,
        height,
      },
      selector: {
        type: 'SvgSelector',
        value: `<svg><polygon points="${points}" /></svg>`,
      },
    },
    body: {
      type: 'FeatureCollection',
      transformation: { type: 'polynomial', options: { order: 1 } },
      features: gcps.map(([x, y]) => ({
        type: 'Feature',
        properties: { resourceCoords: [x, y] },
        geometry: {
          type: 'Point',
          coordinates: [-73.9 + x / 1e6, 40.7 - y / 1e6],
        },
      })),
    },
  };
}

const square: Point[] = [
  [400, 400],
  [3600, 400],
  [3600, 3600],
  [400, 3600],
];

// Detroit 1929 vol 11 p32 from a prototype region mask: clamped to the image, the
// mask's overhang folds onto the bottom edge beside a GCP.
const detroitMask: Point[] = [
  [5023.8, 8097.6],
  [4831.4, 8031.2],
  [4834.4, 7795.0],
  [4154.4, 7795.0],
  [4016.0, 7747.4],
  [3216.6, 7472.3],
  [1088.1, 6746.0],
  [1156.8, 1209.7],
  [1169.7, 360.7],
  [4930.8, 307.8],
  [4917.5, 1171.9],
  [4930.1, 1172.1],
];
const detroitGcps: Point[] = [
  [4822, 7596.2],
  [1784, 178],
];
const detroitP32 = annotation(detroitMask, detroitGcps, [6447, 7795]);

describe('triangulationFailures', () => {
  it('passes a mask with its GCPs inside', () => {
    const gcps: Point[] = [
      [1000, 1000],
      [3000, 1000],
      [2000, 3000],
    ];
    expect(triangulationFailures(annotation(square, gcps))).toEqual([]);
  });

  it('fails a map whose mask runs past the bottom of its image', () => {
    const failures = triangulationFailures(detroitP32);
    expect(failures).toHaveLength(1);
    expect(failures[0].index).toBe(0);
    expect(failures[0].message).toMatch(/intersects/);
  });
});

describe('failureRates', () => {
  it('is zero for a map that draws under every rendering', () => {
    const gcps: Point[] = [
      [1000, 1000],
      [3000, 1000],
      [2000, 3000],
    ];
    expect(failureRates(annotation(square, gcps))).toEqual([0]);
  });

  it('is positive for a map that fails as the viewer draws it', () => {
    const [rate] = failureRates(detroitP32);
    expect(rate).toBeGreaterThan(0);
  });
});
