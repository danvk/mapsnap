import { describe, expect, it } from 'vitest';

import { reportedCount } from './annotations.ts';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations.ts';

const page = (metadata?: { label: string; value: string }[]) =>
  ({
    items: [],
    ...(metadata ? { metadata } : {}),
  }) as unknown as GeorefAnnotationPage;

describe('reportedCount', () => {
  it('reads the run report card off the annotation page', () => {
    // `fit` writes this; Mansfield 1921 cut 42 sheets into 67 images and
    // placed 44 of them, and only the annotation knows the 67 -- the app never
    // sees the images a run declined to place.
    const doc = page([
      { label: 'generated', value: '2026-09-21' },
      { label: 'pages', value: '67' },
      { label: 'placed', value: '44' },
    ]);
    expect(reportedCount(doc, 'pages')).toBe(67);
    expect(reportedCount(doc, 'placed')).toBe(44);
  });

  it('is null when the annotation carries no report', () => {
    // A truth annotation, or one from a run that predates the report card.
    expect(reportedCount(page(), 'pages')).toBeNull();
    expect(
      reportedCount(page([{ label: 'generated', value: 'x' }]), 'pages'),
    ).toBeNull();
  });

  it('is null rather than NaN for a value that is not a number', () => {
    expect(
      reportedCount(page([{ label: 'pages', value: 'many' }]), 'pages'),
    ).toBeNull();
  });
});
