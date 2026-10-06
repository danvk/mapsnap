import { describe, expect, it } from 'vitest';
import { toggledSelection } from './selection';

describe('toggledSelection', () => {
  it('selects what was hit', () => {
    expect(toggledSelection(new Set(), [2])).toEqual(new Set([2]));
    expect(toggledSelection(new Set([1]), [2])).toEqual(new Set([2]));
  });

  it('deselects when the hit is exactly the current selection', () => {
    expect(toggledSelection(new Set([2]), [2])).toEqual(new Set());
  });

  it('keeps a click on nothing a click on nothing', () => {
    expect(toggledSelection(new Set([2]), [])).toEqual(new Set());
  });
});
