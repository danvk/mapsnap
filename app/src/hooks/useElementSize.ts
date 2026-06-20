import { useCallback, useRef, useState } from 'react';

/** The rendered size of an element in CSS pixels. */
export interface ElementSize {
  width: number;
  height: number;
}

/**
 * Track an element's rendered size with a ResizeObserver.
 *
 * Returns a callback ref to attach to the element and its current `offsetWidth`
 * / `offsetHeight`. Because it is a callback ref, observation is set up whenever
 * the element attaches — including elements that mount conditionally after the
 * initial render. The size updates whenever the element is resized.
 */
export function useElementSize<T extends HTMLElement>(): [
  (node: T | null) => void,
  ElementSize,
] {
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 });
  const observerRef = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: T | null) => {
    observerRef.current?.disconnect();
    observerRef.current = null;
    if (!node) return;
    const update = () =>
      setSize({ width: node.offsetWidth, height: node.offsetHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    observerRef.current = observer;
  }, []);

  return [ref, size];
}
