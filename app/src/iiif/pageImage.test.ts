import { describe, expect, it } from 'vitest';
import { pageImageFromParam, pageImageParam } from './pageImage';

describe('page image URL parameter', () => {
  it('round-trips the alternate images and omits the default sheet', () => {
    expect(pageImageParam('page')).toBeNull();
    expect(pageImageParam('region')).toBe('region');
    expect(pageImageFromParam(pageImageParam('roadprob'))).toBe('roadprob');
  });

  it('reads anything unknown as the sheet', () => {
    expect(pageImageFromParam(null)).toBe('page');
    expect(pageImageFromParam('heatmap')).toBe('page');
  });
});
