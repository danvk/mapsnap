import { describe, expect, it } from 'vitest';
import {
  consistentGcps,
  keymapAnnotation,
  MIN_KEYMAP_GCPS,
  smoothedTargets,
  type KeymapGcp,
} from './keymapAnnotation.ts';

const SERVICE = 'http://localhost:8182/iiif/new_orleans/raw/p0.jpg';
const corners = [
  [-90.1158, 29.9432],
  [-90.0872, 29.9745],
  [-90.0431, 29.9458],
  [-90.0717, 29.9145],
];

// `count` inliers spread along a diagonal, far enough apart to stay distinct.
function inliers(count: number): KeymapGcp[] {
  return Array.from({ length: count }, (unused, i) => ({
    x: 100 + i * 37,
    y: 200 + i * 41,
    lon: -90.1 + i * 0.001,
    lat: 29.94 + i * 0.001,
    inlier: true,
  }));
}

function georefWith(count: number) {
  return {
    width: 6397,
    height: 7795,
    corners,
    intersections: inliers(count),
  };
}

describe('consistentGcps', () => {
  it('keeps a pixel read once', () => {
    expect(consistentGcps(inliers(3))).toHaveLength(3);
  });

  it('drops a whole group whose readings disagree beyond the conflict bar', () => {
    const conflicting: KeymapGcp[] = [
      { x: 10, y: 20, lon: -90.1, lat: 29.94 },
      { x: 10, y: 20, lon: -90.09, lat: 29.94 }, // ~2900 ft away
      { x: 30, y: 40, lon: -90.05, lat: 29.95 },
    ];
    expect(consistentGcps(conflicting).map((gcp) => gcp.x)).toEqual([30]);
  });

  it('keeps one reading of a group that agrees within the bar', () => {
    const agreeing: KeymapGcp[] = [
      { x: 10, y: 20, lon: -90.1, lat: 29.94 },
      { x: 10, y: 20, lon: -90.10002, lat: 29.94001 }, // a few feet
    ];
    expect(consistentGcps(agreeing)).toHaveLength(1);
  });
});

describe('keymapAnnotation', () => {
  it('warps by thin-plate spline through the sheet own GCPs', () => {
    const page = keymapAnnotation(georefWith(60), SERVICE, 'keymap-p0');
    const item = page!.items[0];
    const body = item.body as unknown as {
      transformation: { type: string };
      features: { properties: { resourceCoords: number[] } }[];
    };
    expect(body.transformation).toEqual({ type: 'thinPlateSpline' });
    expect(body.features).toHaveLength(60);
    expect(body.features[0].properties.resourceCoords).toEqual([100, 200]);
    expect(item.target!.source).toEqual({
      id: SERVICE,
      type: 'ImageService3',
      width: 6397,
      height: 7795,
    });
    // The whole sheet is the mask: the key map is drawn margins and all.
    expect(
      (item.target as unknown as { selector: { value: string } }).selector
        .value,
    ).toContain('points="0,0 6397,0 6397,7795 0,7795"');
  });

  it('hands allmaps the smoothed model targets, not the raw OSM points', () => {
    // Noisy GCPs: the smoothed spline does not pass through them, so the
    // annotation's targets differ from the inputs; an exact spline through
    // the targets is then the pipeline's smoothed surface.
    const noisy = inliers(40).map((gcp, i) => ({
      ...gcp,
      lat: gcp.lat + (i % 3 === 0 ? 0.0004 : 0),
    }));
    const page = keymapAnnotation(
      { width: 6397, height: 7795, intersections: noisy },
      SERVICE,
      'id',
    );
    const body = page!.items[0].body as unknown as {
      features: { geometry: { coordinates: number[] } }[];
    };
    const moved = body.features.filter(
      (f, i) => Math.abs(f.geometry.coordinates[1] - noisy[i].lat) > 1e-6,
    );
    expect(moved.length).toBeGreaterThan(30);
    // An affine field is left exactly alone, at any smoothing.
    const clean = smoothedTargets(inliers(40));
    clean.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(inliers(40)[i].lon, 7);
      expect(lat).toBeCloseTo(inliers(40)[i].lat, 7);
    });
  });

  it('falls back to the affine corners when the sheet has too few GCPs', () => {
    const page = keymapAnnotation(
      georefWith(MIN_KEYMAP_GCPS - 1),
      SERVICE,
      'keymap-p0',
    );
    const body = page!.items[0].body as unknown as {
      transformation: { type: string; options: { order: number } };
      features: {
        properties: { resourceCoords: number[] };
        geometry: { coordinates: number[] };
      }[];
    };
    expect(body.transformation).toEqual({
      type: 'polynomial',
      options: { order: 1 },
    });
    // Pixel corners in the georef's own order: TL, TR, BR, BL.
    expect(body.features.map((f) => f.properties.resourceCoords)).toEqual([
      [0, 0],
      [6397, 0],
      [6397, 7795],
      [0, 7795],
    ]);
    expect(body.features[1].geometry.coordinates).toEqual(corners[1]);
  });

  it('is null without dimensions, or with too few GCPs and no corners', () => {
    expect(keymapAnnotation({}, SERVICE, 'id')).toBeNull();
    expect(
      keymapAnnotation(
        { width: 10, height: 10, intersections: [] },
        SERVICE,
        'id',
      ),
    ).toBeNull();
  });
});
