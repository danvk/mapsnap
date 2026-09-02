import { describe, expect, it } from 'vitest';
import { keymapUnderlays, underlayFromParam, underlayParam } from './underlay';
import type { KeymapInfo } from '../../server/api';

const corners: [number, number][] = [
  [-90.1158, 29.9432],
  [-90.0872, 29.9745],
  [-90.0431, 29.9458],
  [-90.0717, 29.9145],
];

const service = 'http://localhost:8182/iiif/new_orleans_la_1896_vol_2/raw';
const keymaps: KeymapInfo[] = [
  {
    stem: 'p0',
    hasRegions: true,
    hasGeoref: true,
    hasRoadprob: true,
    corners,
    imageService: `${service}/p0.jpg`,
    roadprobService: `${service}/p0.roadprob.png`,
  },
  {
    stem: 'pb',
    hasRegions: false,
    hasGeoref: true,
    hasRoadprob: false,
    corners,
    imageService: `${service}/pb.png`,
  },
  {
    stem: 'pz',
    hasRegions: false,
    hasGeoref: false,
    hasRoadprob: true,
    imageService: `${service}/pz.jpg`,
    roadprobService: `${service}/pz.roadprob.png`,
  },
];

describe('keymapUnderlays', () => {
  it('draws every georeferenced key map as its sheet through the IIIF service', () => {
    const underlays = keymapUnderlays(keymaps, 'image');
    expect(underlays.map((u) => u.id)).toEqual(['p0', 'pb']);
    expect(underlays[0].url).toBe(
      `${service}/p0.jpg/full/!2400,2400/0/default.jpg`,
    );
    expect(underlays[0].corners).toEqual(corners);
  });

  it('draws only key maps with a P(road) map in roadprob mode', () => {
    const underlays = keymapUnderlays(keymaps, 'roadprob');
    expect(underlays.map((u) => u.id)).toEqual(['p0']);
    expect(underlays[0].url).toBe(
      `${service}/p0.roadprob.png/full/!2400,2400/0/default.png`,
    );
  });

  it('draws nothing when off', () => {
    expect(keymapUnderlays(keymaps, 'off')).toEqual([]);
  });
});

describe('underlay URL parameter', () => {
  it('round-trips the two on modes and treats anything else as off', () => {
    expect(underlayFromParam('image')).toBe('image');
    expect(underlayFromParam('roadprob')).toBe('roadprob');
    expect(underlayFromParam(null)).toBe('off');
    expect(underlayFromParam('yes')).toBe('off');
    expect(underlayParam('off')).toBeNull();
    expect(underlayParam('roadprob')).toBe('roadprob');
  });
});
