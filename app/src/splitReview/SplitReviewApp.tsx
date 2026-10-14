import { useEffect, useMemo, useRef, useState } from 'react';
import './splitReview.css';
import { isTypingTarget, nextOpacityStep } from '../keyboard';
import {
  armOutcome,
  initialIndex,
  pageImageUrl,
  reviewSummary,
  stepIndex,
} from './review';
import { SplitPane } from './SplitPane';
import type { SplitReview } from './types';

/**
 * Split review: step through pages comparing two splitter arms (A/B) against
 * OIM's truth panels.
 *
 * The review is a `review.json` that scripts/split_review.py writes, named by the
 * `?review=` parameter (e.g. `?review=data/split-review/test-100/review.json`).
 * The selected page goes in `?page=`, so a reload or a shared link keeps it.
 * j/k (or the arrow keys) step through the list, t toggles the truth outlines
 * and p cycles the panel opacity.
 */
export function SplitReviewApp() {
  const params = new URLSearchParams(window.location.search);
  const reviewUrl = params.get('review');
  const [review, setReview] = useState<SplitReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [showTruth, setShowTruth] = useState(true);
  const [opacity, setOpacity] = useState(100);
  const selectedRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    if (!reviewUrl) return;
    fetch(reviewUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<SplitReview>;
      })
      .then((data) => {
        setReview(data);
        setIndex(initialIndex(data, params.get('page')));
      })
      .catch((e: unknown) => setError(`${reviewUrl}: ${String(e)}`));
    // The URL is read once; ?page= is kept in sync below, not re-read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewUrl]);

  const pageCount = review?.pages.length ?? 0;
  const page = review?.pages[index];

  // Keep ?page= in step with the selection, without adding history entries.
  useEffect(() => {
    if (!page) return;
    const url = new URL(window.location.href);
    url.searchParams.set('page', page.name);
    window.history.replaceState(null, '', url);
  }, [page]);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest' });
  }, [index]);

  // Warm the cache with the next page's image, so stepping through is instant.
  useEffect(() => {
    const next = review?.pages[index + 1];
    if (!next || !reviewUrl) return;
    new Image().src = pageImageUrl(reviewUrl, next.image, window.location.href);
  }, [review, index, reviewUrl]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      if (isTypingTarget(event.target) || event.metaKey || event.ctrlKey) {
        return;
      }
      if (event.key === 'j' || event.key === 'ArrowDown') {
        event.preventDefault();
        setIndex((i) => stepIndex(i, 1, pageCount));
      } else if (event.key === 'k' || event.key === 'ArrowUp') {
        event.preventDefault();
        setIndex((i) => stepIndex(i, -1, pageCount));
      } else if (event.key === 't') {
        setShowTruth((shown) => !shown);
      } else if (event.key === 'p') {
        setOpacity(nextOpacityStep);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [pageCount]);

  const summary = useMemo(() => review && reviewSummary(review), [review]);

  if (!reviewUrl) {
    return (
      <p className="sr-message">
        Name a review: <code>?review=data/split-review/…/review.json</code>{' '}
        (written by <code>scripts/split_review.py</code>).
      </p>
    );
  }
  if (error) return <p className="sr-message sr-bad">{error}</p>;
  if (!review || !summary) return <p className="sr-message">Loading…</p>;

  return (
    <div className="sr-app">
      <div className="sr-sidebar">
        <h2>{review.title}</h2>
        <table className="sr-summary">
          <tbody>
            {review.arms.map((label, arm) => (
              <tr key={arm}>
                <th>{arm === 0 ? 'A' : 'B'}</th>
                <td>{label}</td>
                <td title="pages with the truth's panel count">
                  {summary[arm]!.right}/{pageCount} right
                </td>
                <td title="mean matched IoU">
                  {summary[arm]!.meanIou.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <ul className="sr-list">
          {review.pages.map((p, i) => {
            const a = armOutcome(p, 0);
            const b = armOutcome(p, 1);
            const delta = b.iou - a.iou;
            return (
              <li
                key={p.name}
                ref={i === index ? selectedRef : undefined}
                className={i === index ? 'selected' : undefined}
                onClick={() => setIndex(i)}
                title={p.title}
              >
                <span className="sr-name">{p.name}</span>
                <span className="sr-marks">
                  <span className={a.right ? 'sr-ok' : 'sr-bad'}>
                    A{a.right ? '✓' : '✗'}
                  </span>
                  <span className={b.right ? 'sr-ok' : 'sr-bad'}>
                    B{b.right ? '✓' : '✗'}
                  </span>
                  <span
                    className={`sr-delta ${delta > 0 ? 'sr-ok' : delta < 0 ? 'sr-bad' : ''}`}
                    title="IoU, B minus A"
                  >
                    {delta >= 0 ? '+' : '−'}
                    {Math.abs(delta).toFixed(2)}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      {page && (
        <div className="sr-main">
          <div className="sr-toolbar">
            <span className="sr-title">
              {page.title || page.item} · {page.page}
            </span>
            <span className="sr-muted">
              {page.name} · {index + 1}/{pageCount} ·{' '}
              {page.label === 'split'
                ? `OIM split into ${page.truth.length}`
                : 'OIM left whole'}
            </span>
            <label>
              <input
                type="checkbox"
                checked={showTruth}
                onChange={(e) => setShowTruth(e.target.checked)}
              />
              Truth (t)
            </label>
            <label title="Press p to cycle 0/50/100%">
              <input
                type="range"
                min={0}
                max={100}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              Opacity (p): {opacity}%
            </label>
            <span className="sr-muted">j/k: next/previous</span>
          </div>
          <div className="sr-panes">
            {([0, 1] as const).map((arm) => (
              <SplitPane
                key={arm}
                side={arm === 0 ? 'A' : 'B'}
                armLabel={review.arms[arm]}
                outcome={armOutcome(page, arm)}
                imageUrl={pageImageUrl(
                  reviewUrl,
                  page.image,
                  window.location.href,
                )}
                width={page.width}
                height={page.height}
                panels={page.arms[arm].panels}
                truth={page.truth}
                showTruth={showTruth}
                opacity={opacity}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
