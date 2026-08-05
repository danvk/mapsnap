import { describe, expect, it } from 'vitest';
import { isSafeSegment, isSafeVolume, MAX_VOLUME_DEPTH } from './volumePaths';

describe('isSafeSegment', () => {
  it('accepts an ordinary directory name', () => {
    expect(isSafeSegment('detroit_mich_1929_vol_11')).toBe(true);
    expect(isSafeSegment('brooklyn_1904-1908')).toBe(true);
  });

  it('rejects anything that could escape the data directory', () => {
    expect(isSafeSegment('')).toBe(false);
    expect(isSafeSegment('..')).toBe(false);
    expect(isSafeSegment('a/b')).toBe(false);
    expect(isSafeSegment('a\\b')).toBe(false);
    expect(isSafeSegment('../etc')).toBe(false);
    expect(isSafeSegment('.')).toBe(false);
  });
});

describe('isSafeVolume', () => {
  it('accepts a single-directory volume', () => {
    expect(isSafeVolume('detroit_mich_1929_vol_11')).toBe(true);
  });

  it('accepts a subvolume of a multi-volume atlas', () => {
    expect(isSafeVolume('brooklyn_1904-1908/vol13')).toBe(true);
    expect(isSafeVolume('queens_1950/vol2')).toBe(true);
  });

  it('rejects a path deeper than a volume', () => {
    // A volume's own subdirectories are not volumes: allowing this would let
    // ?volume=<vol>/raw address the key-map directory as a volume.
    expect(isSafeVolume('brooklyn_1904-1908/vol13/raw')).toBe(false);
    expect(isSafeVolume('a/b/c/d')).toBe(false);
  });

  it('rejects escapes in any segment', () => {
    expect(isSafeVolume('..')).toBe(false);
    expect(isSafeVolume('brooklyn_1904-1908/..')).toBe(false);
    expect(isSafeVolume('../brooklyn_1904-1908')).toBe(false);
    expect(isSafeVolume('brooklyn_1904-1908/.')).toBe(false);
    // "a/" is depth 2 with an empty second segment and must still be rejected.
    expect(isSafeVolume('a/')).toBe(false);
    expect(isSafeVolume('')).toBe(false);
  });

  it('agrees with the documented depth limit', () => {
    expect(MAX_VOLUME_DEPTH).toBe(2);
    expect(isSafeVolume(Array(MAX_VOLUME_DEPTH).fill('v').join('/'))).toBe(
      true,
    );
    expect(
      isSafeVolume(
        Array(MAX_VOLUME_DEPTH + 1)
          .fill('v')
          .join('/'),
      ),
    ).toBe(false);
  });
});
