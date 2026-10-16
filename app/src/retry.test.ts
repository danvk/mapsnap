import { describe, expect, it } from 'vitest';
import { retryOnNetworkError } from './retry';

/** An attempt that fails with each error in turn, then succeeds. */
function failingThen(errors: Error[]): {
  attempt: () => Promise<string>;
  calls: () => number;
} {
  let calls = 0;
  return {
    attempt: async () => {
      const error = errors[calls++];
      if (error) throw error;
      return 'ok';
    },
    calls: () => calls,
  };
}

describe('retryOnNetworkError', () => {
  it('retries a request that never reached the server', async () => {
    const { attempt, calls } = failingThen([
      new TypeError('Failed to fetch'),
      new TypeError('Failed to fetch'),
    ]);
    await expect(retryOnNetworkError(attempt, { delayMs: 0 })).resolves.toBe(
      'ok',
    );
    expect(calls()).toBe(3);
  });

  it('gives up after the last try with the network error', async () => {
    const { attempt, calls } = failingThen([
      new TypeError('Failed to fetch'),
      new TypeError('Failed to fetch'),
      new TypeError('Failed to fetch'),
    ]);
    await expect(retryOnNetworkError(attempt, { delayMs: 0 })).rejects.toThrow(
      TypeError,
    );
    expect(calls()).toBe(3);
  });

  it('passes an HTTP error straight through', async () => {
    const { attempt, calls } = failingThen([new Error('GET /x: HTTP 404')]);
    await expect(retryOnNetworkError(attempt, { delayMs: 0 })).rejects.toThrow(
      'HTTP 404',
    );
    expect(calls()).toBe(1);
  });
});
