import { describe, expect, it, vi } from 'vitest';

import { firstImage, roadProbCandidates } from './roadProb';

describe('roadProbCandidates', () => {
  it('offers the sidecar first and the pre-#354 PNG second', () => {
    expect(roadProbCandidates('data/queens_ny_1898', 'p17')).toEqual([
      'data/queens_ny_1898/p17.roadprob.jpg',
      'data/queens_ny_1898/artifacts/edge_join/roadprob/p17.png',
    ]);
  });

  it('keeps a split panel its own map', () => {
    // split cuts the parent's P(road) map into the panels', so p20__3 has one.
    expect(roadProbCandidates('data/madison_in_1904', 'p20__3')[0]).toBe(
      'data/madison_in_1904/p20__3.roadprob.jpg',
    );
  });
});

describe('firstImage', () => {
  const headed = (images: string[]) =>
    vi.fn(async (url: string) => ({
      ok: images.includes(url),
      headers: {
        get: () => (images.includes(url) ? 'image/jpeg' : 'text/html'),
      },
    })) as unknown as typeof fetch;

  it('takes the sidecar when it is there', async () => {
    vi.stubGlobal('fetch', headed(['a.jpg', 'b.png']));
    expect(await firstImage(['a.jpg', 'b.png'])).toBe('a.jpg');
  });

  it('falls back to the old PNG for a volume that predates the sidecars', async () => {
    vi.stubGlobal('fetch', headed(['b.png']));
    expect(await firstImage(['a.jpg', 'b.png'])).toBe('b.png');
  });

  it('is null when neither is on disk, rather than a broken image', async () => {
    vi.stubGlobal('fetch', headed([]));
    expect(await firstImage(['a.jpg', 'b.png'])).toBeNull();
  });

  it('treats a dev-server HTML fallback as absent', async () => {
    // Vite answers an unknown path with index.html, 200 and all.
    vi.stubGlobal('fetch', headed(['a.jpg']));
    expect(await firstImage(['missing.jpg'])).toBeNull();
  });
});
