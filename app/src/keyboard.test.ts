import { describe, expect, it } from 'vitest';
import { nextOpacityStep } from './keyboard';

describe('nextOpacityStep', () => {
  it('cycles 0 -> 50 -> 100 -> 0', () => {
    expect(nextOpacityStep(0)).toBe(50);
    expect(nextOpacityStep(50)).toBe(100);
    expect(nextOpacityStep(100)).toBe(0);
  });

  it('starts the cycle over from a slider value between stops', () => {
    expect(nextOpacityStep(73)).toBe(0);
  });
});
