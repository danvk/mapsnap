import type { HTTPVerb } from 'crosswalk/dist/api-spec';

/**
 * `fetch` adapter for crosswalk's `typedApi`.
 *
 * crosswalk's built-in fetcher always sends a JSON body, but the browser rejects
 * a body on GET/HEAD ("Request with GET/HEAD method cannot have body"). This
 * omits the body for bodyless verbs and throws on a non-2xx response so callers
 * see a rejected promise rather than a mistyped payload.
 */
export async function jsonFetch(
  url: string,
  method: HTTPVerb,
  payload: unknown,
): Promise<unknown> {
  const hasBody = method !== 'get' && method !== 'delete';
  const response = await fetch(url, {
    method,
    headers: hasBody
      ? { Accept: 'application/json', 'Content-Type': 'application/json' }
      : { Accept: 'application/json' },
    body: hasBody ? JSON.stringify(payload) : undefined,
  });
  if (!response.ok) {
    throw new Error(`${method.toUpperCase()} ${url}: HTTP ${response.status}`);
  }
  return response.json();
}
