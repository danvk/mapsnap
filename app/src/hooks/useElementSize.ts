import { useEffect, useRef, useState } from 'react';

/** The rendered size of an element in CSS pixels. */
export interface ElementSize {
  width: number;
  height: number;
}

/**
 * Track an element's rendered size with a ResizeObserver.
 *
 * Returns a ref to attach to the element and its current `offsetWidth` /
 * `offsetHeight`. The size updates whenever the element is resized.
 */
export function useElementSize<T extends HTMLElement>(): [
  React.RefObject<T | null>,
  ElementSize,
] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () =>
      setSize({ width: element.offsetWidth, height: element.offsetHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}
