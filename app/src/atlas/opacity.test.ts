import { describe, expect, it } from 'vitest';

import { nextOpacity } from './opacity.ts';

describe('nextOpacity', () => {
  it('steps 100 -> 50 -> 0 -> 100', () => {
    expect(nextOpacity(100)).toBe(50);
    expect(nextOpacity(50)).toBe(0);
    expect(nextOpacity(0)).toBe(100);
  });

  it('starts the cycle over from a value between steps', () => {
    expect(nextOpacity(73)).toBe(100);
  });
});
