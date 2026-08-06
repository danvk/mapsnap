import { describe, expect, it } from 'vitest';
import { runArtifactDir, runArtifactStems } from './runArtifacts.ts';

describe('runArtifactDir', () => {
  it('maps a volume-root annotation to its artifacts directory', () => {
    expect(
      runArtifactDir('detroit_mich_1929_vol_11/2026-07-30-full.iiif.json'),
    ).toBe('detroit_mich_1929_vol_11/artifacts/2026-07-30-full');
  });

  it('returns the directory an annotation already sits in', () => {
    // Opening the copy inside the run's own directory is its own answer; going
    // up to the volume and back down would resolve to the same place, but only
    // by luck of the tag matching the folder name.
    expect(
      runArtifactDir(
        'detroit_mich_1929_vol_11/artifacts/2026-07-30-full/2026-07-30-full.iiif.json',
      ),
    ).toBe('detroit_mich_1929_vol_11/artifacts/2026-07-30-full');
  });

  it('handles nested volume paths', () => {
    expect(runArtifactDir('brooklyn_1904-1908/vol13/b175.iiif.json')).toBe(
      'brooklyn_1904-1908/vol13/artifacts/b175',
    );
  });

  it('returns null for a non-annotation path', () => {
    expect(
      runArtifactDir('detroit_mich_1929_vol_11/p1.georef.json'),
    ).toBeNull();
    expect(runArtifactDir('main.iiif.json')).toBeNull();
  });
});

describe('runArtifactStems', () => {
  it('counts every per-page sidecar kind, not just the plain georef', () => {
    // p64 was demoted then rescued, so it has no plain georef.json at all.
    expect(
      runArtifactStems([
        'p1.georef.json',
        'p64.georef-osm.json',
        'p64.georef-contradicted.json',
        'p64.streets.json',
        'p19.georef-nofit.json',
        'p24.streets.json',
      ]),
    ).toEqual(['p1', 'p19', 'p24', 'p64']);
  });

  it('deduplicates a page with several sidecars', () => {
    expect(
      runArtifactStems([
        'p5.georef.json',
        'p5.georef-osm.json',
        'p5.streets.json',
      ]),
    ).toEqual(['p5']);
  });

  it('handles split panels and lettered stems', () => {
    expect(
      runArtifactStems([
        'p12__1.georef.json',
        'p12__2.streets.json',
        'p33B.georef-1gcp.json',
      ]),
    ).toEqual(['p12__1', 'p12__2', 'p33B']);
  });

  it('ignores run-level files, including a tag that starts with p', () => {
    // A bare `^(p[^.]+)\.` pattern would read "prod" out of prod.iiif.json.
    expect(
      runArtifactStems([
        'manifest.json',
        'prod.iiif.json',
        'prod.txt',
        'p7.txt',
        'p7.georef.json',
      ]),
    ).toEqual(['p7']);
  });
});
