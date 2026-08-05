import { describe, expect, it } from 'vitest';
import { runArtifactDir } from './runArtifacts.ts';

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
