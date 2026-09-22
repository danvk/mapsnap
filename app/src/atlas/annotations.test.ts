import { describe, expect, it } from 'vitest';

import { inParallel, stemFromLabel, withPageMetadata } from './annotations.ts';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations.ts';

describe('stemFromLabel', () => {
  it('takes the page off a published label', () => {
    expect(
      stemFromLabel('Chicago, Illinois | 1950 | sanborn01790_085 p100W'),
    ).toBe('p100W');
  });

  it('folds a split panel back into its stem', () => {
    // A published split labels its panel in brackets; the on-disk stem, which
    // is what the geometry code matches on, joins them with __.
    expect(
      stemFromLabel('Saint Louis, Missouri | 1916 | sanborn04858_015 p66 [1]'),
    ).toBe('p66__1');
  });

  it('is null for a label that names no page', () => {
    expect(stemFromLabel('Chicago, Illinois | 1950 | key map')).toBeNull();
    expect(stemFromLabel(undefined)).toBeNull();
    expect(stemFromLabel('')).toBeNull();
  });
});

describe('withPageMetadata', () => {
  const page = (label: string, metadata?: { label: string; value: string }[]) =>
    ({
      items: [{ label, ...(metadata ? { metadata } : {}) }],
    }) as unknown as GeorefAnnotationPage;

  it('adds the page entry a published annotation does not carry', () => {
    const out = withPageMetadata(page('X | 1950 | sanborn01_001 p12'));
    expect(out.items?.[0]?.metadata).toContainEqual({
      label: 'page',
      value: 'p12',
    });
  });

  it('keeps entries the annotation already has', () => {
    const out = withPageMetadata(
      page('X | 1950 | sanborn01_001 p12', [{ label: 'streets', value: '4' }]),
    );
    expect(out.items?.[0]?.metadata).toHaveLength(2);
  });

  it('leaves an annotation that already names its pages alone', () => {
    const existing = [{ label: 'page', value: 'p9' }];
    const out = withPageMetadata(page('whatever p12', existing));
    expect(out.items?.[0]?.metadata).toEqual(existing);
  });

  it('does not mutate its argument', () => {
    // The same object is handed to Allmaps; editing it in place would make the
    // annotation depend on whether the panel had read it first.
    const input = page('X | 1950 | sanborn01_001 p12');
    withPageMetadata(input);
    expect(input.items?.[0]?.metadata).toBeUndefined();
  });
});

describe('inParallel', () => {
  it('keeps input order however the work finishes', async () => {
    const delays = [30, 1, 20, 2, 10];
    const out = await inParallel(delays, 2, async (ms) => {
      await new Promise((resolve) => setTimeout(resolve, ms));
      return ms;
    });
    expect(out).toEqual(delays);
  });

  it('runs no more than the given width at once', async () => {
    let live = 0;
    let peak = 0;
    await inParallel([...Array(12).keys()], 3, async () => {
      peak = Math.max(peak, ++live);
      await new Promise((resolve) => setTimeout(resolve, 1));
      live--;
    });
    expect(peak).toBeLessThanOrEqual(3);
  });

  it('has nothing to do for an empty list', async () => {
    expect(await inParallel([], 4, async () => 1)).toEqual([]);
  });
});
