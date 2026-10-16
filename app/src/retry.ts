/**
 * Retry a request that failed to reach the server at all.
 *
 * `fetch` rejects with a TypeError ("Failed to fetch") when the request never
 * completed -- a connection the dev proxy or the local server dropped while a
 * volume's page images were loading. That is worth another try. An HTTP error
 * is an answer (jsonFetch throws a plain Error for one) and is passed straight
 * through: a 404 from an older server should fall back at once.
 *
 * Opening Oakland 1903's corpus-v1 run from a deep link lost its
 * /iiif-api/run-artifacts request this way, and the page panel then said the
 * run had no artifact directory when it had 173 pages of sidecars.
 */
export async function retryOnNetworkError<T>(
  attempt: () => Promise<T>,
  { tries = 3, delayMs = 400 }: { tries?: number; delayMs?: number } = {},
): Promise<T> {
  for (let n = 1; ; n++) {
    try {
      return await attempt();
    } catch (error) {
      if (!(error instanceof TypeError) || n >= tries) throw error;
      await new Promise((resolve) => setTimeout(resolve, delayMs * n));
    }
  }
}
