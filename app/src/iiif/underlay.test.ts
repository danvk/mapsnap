import { describe, expect, it } from 'vitest';
import {
  keymapUnderlays,
  underlayImageFromParam,
  underlayImageParam,
} from './underlay';
import type { KeymapInfo } from '../../server/api';

const corners: [number, number][] = [
  [-90.1158, 29.9432],
  [-90.0872, 29.9745],
  [-90.0431, 29.9458],
  [-90.0717, 29.9145],
];

const keymaps: KeymapInfo[] = [
  {
    stem: 'p0',
    sidecarDir: 'raw',
    hasRegions: true,
    hasGeoref: true,
    hasRoadprob: true,
    corners,
  },
  {
    stem: 'pb',
    sidecarDir: 'raw',
    hasRegions: false,
    hasGeoref: true,
    hasRoadprob: false,
    corners,
  },
  // No georef: nothing to warp it by, in either mode.
  {
    stem: 'pz',
    sidecarDir: 'raw',
    hasRegions: false,
    hasGeoref: false,
    hasRoadprob: true,
  },
];

describe('keymapUnderlays', () => {
  it('draws every georeferenced key map from its annotation', () => {
    const underlays = keymapUnderlays(
      'new_orleans_la_1896_vol_2',
      keymaps,
      'sheet',
    );
    expect(underlays.map((u) => u.id)).toEqual(['p0', 'pb']);
    expect(underlays[0].annotationUrl).toBe(
      '/iiif-api/keymap-annotation?volume=new_orleans_la_1896_vol_2&stem=p0&image=sheet',
    );
  });

  it('draws only key maps with a P(road) map in roadprob mode', () => {
    const underlays = keymapUnderlays('queens_1950/vol2', keymaps, 'roadprob');
    expect(underlays.map((u) => u.id)).toEqual(['p0']);
    expect(underlays[0].annotationUrl).toBe(
      '/iiif-api/keymap-annotation?volume=queens_1950%2Fvol2&stem=p0&image=roadprob',
    );
  });
});

describe('keymapUnderlays with a mirror run', () => {
  it("asks for the run's key map (#554)", () => {
    const underlays = keymapUnderlays(
      'queens_ny_1950_vol_1',
      keymaps,
      'sheet',
      'runs/corpus-v1',
    );
    expect(underlays[0].annotationUrl).toBe(
      '/iiif-api/keymap-annotation?volume=queens_ny_1950_vol_1&stem=p0&image=sheet&run=runs%2Fcorpus-v1',
    );
  });
});

describe('underlay image URL parameter', () => {
  it('round-trips roadprob and treats anything else as the sheet', () => {
    expect(underlayImageFromParam('roadprob')).toBe('roadprob');
    expect(underlayImageFromParam('sheet')).toBe('sheet');
    expect(underlayImageFromParam(null)).toBe('sheet');
    expect(underlayImageFromParam('yes')).toBe('sheet');
    expect(underlayImageParam('sheet')).toBeNull();
    expect(underlayImageParam('roadprob')).toBe('roadprob');
  });
});
