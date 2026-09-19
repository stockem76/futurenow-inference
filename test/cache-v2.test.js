'use strict';
/**
 * cache-v2.test.js — Tests for new cache.js v2 APIs.
 *
 * Tests: getMany(), setMany(), contentHash(), getOrEmbed().
 */

const { ok, equal, near } = global._assert;

const cache = require('../src/cache');

function vec(values) { return new Float32Array(values); }

function vecEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 1e-6) return false;
  }
  return true;
}

module.exports = async function () {

  cache._setConfig({ hotMax: 20, ttlMs: 5000 });

  const v1 = vec([0.1, 0.2, 0.3]);
  const v2 = vec([0.4, 0.5, 0.6]);
  const v3 = vec([0.7, 0.8, 0.9]);

  // ── contentHash ───────────────────────────────────────────────────────────

  const h1 = cache.contentHash('react developer');
  const h2 = cache.contentHash('react developer');   // same text
  const h3 = cache.contentHash('python engineer');   // different text
  const h4 = cache.contentHash('');
  const h5 = cache.contentHash(null);

  equal(h1, h2,                                           'contentHash: deterministic for same text');
  ok(h1 !== h3,                                           'contentHash: different text → different hash');
  ok(typeof h1 === 'string' && h1.length === 8,           'contentHash: returns 8-char string');
  ok(typeof h4 === 'string',                              'contentHash: empty string → valid hash');
  ok(typeof h5 === 'string',                              'contentHash: null → valid hash');
  equal(h4, h5,                                           'contentHash: empty and null → same hash');

  // Hash should only contain hex chars
  ok(/^[0-9a-f]{8}$/.test(h1),                           'contentHash: valid hex string');

  // ── getMany ───────────────────────────────────────────────────────────────

  cache.flush();
  cache.set('gm1', v1, 'hot');
  cache.set('gm2', v2, 'hot');
  // gm3 not set

  const many = cache.getMany(['gm1', 'gm2', 'gm3']);
  ok(many instanceof Map,                                  'getMany: returns Map');
  equal(many.size, 3,                                     'getMany: Map has entry for every key');
  ok(vecEqual(many.get('gm1'), v1),                       'getMany: gm1 value correct');
  ok(vecEqual(many.get('gm2'), v2),                       'getMany: gm2 value correct');
  equal(many.get('gm3'), null,                            'getMany: missing key → null');

  // Empty array → empty Map
  const emptyMany = cache.getMany([]);
  ok(emptyMany instanceof Map && emptyMany.size === 0,    'getMany: empty array → empty Map');

  // ── setMany ───────────────────────────────────────────────────────────────

  cache.flush();
  cache.setMany([
    { key: 'sm1', vec: v1 },
    { key: 'sm2', vec: v2 },
    { key: 'sm3', vec: v3 },
  ], 'hot');

  ok(vecEqual(cache.get('sm1'), v1),                      'setMany: sm1 stored correctly');
  ok(vecEqual(cache.get('sm2'), v2),                      'setMany: sm2 stored correctly');
  ok(vecEqual(cache.get('sm3'), v3),                      'setMany: sm3 stored correctly');

  // setMany with empty array → no error
  cache.setMany([], 'hot');
  ok(vecEqual(cache.get('sm1'), v1),                      'setMany: empty call preserves existing');

  // setMany with ttl tier
  cache.flush();
  cache.setMany([{ key: 'smttl', vec: v1 }], 'ttl');
  ok(vecEqual(cache.get('smttl'), v1),                    'setMany: ttl tier stored correctly');

  // ── getOrEmbed ────────────────────────────────────────────────────────────

  cache.flush();

  let embedCallCount = 0;
  const fakeEmbedFn = async (text) => {
    embedCallCount++;
    return vec([0.1 * embedCallCount, 0.2, 0.3]);
  };

  const text1 = 'react developer experience';

  // First call should invoke embedFn
  const r1 = await cache.getOrEmbed(text1, fakeEmbedFn);
  ok(r1 instanceof Float32Array,                          'getOrEmbed: returns Float32Array');
  equal(embedCallCount, 1,                                'getOrEmbed: embedFn called once on miss');

  // Second call with same text should hit cache
  const r2 = await cache.getOrEmbed(text1, fakeEmbedFn);
  equal(embedCallCount, 1,                                'getOrEmbed: embedFn NOT called on hit');
  ok(vecEqual(r1, r2),                                    'getOrEmbed: cached result matches original');

  // Different text → new embedFn call
  await cache.getOrEmbed('typescript engineer', fakeEmbedFn);
  equal(embedCallCount, 2,                                'getOrEmbed: different text → new embedFn call');

  // embedFn returning null → return null, don't cache
  const nullEmbedFn = async () => null;
  const nullResult = await cache.getOrEmbed('no embed available', nullEmbedFn);
  equal(nullResult, null,                                 'getOrEmbed: null from embedFn → returns null');

  // Subsequent call on same text with null embedFn should call it again (not cached)
  let nullCallCount = 0;
  const countingNullFn = async () => { nullCallCount++; return null; };
  await cache.getOrEmbed('no embed available', countingNullFn);
  await cache.getOrEmbed('no embed available', countingNullFn);
  equal(nullCallCount, 2,                                 'getOrEmbed: null result not cached');

  // contentHash used as cache key (verify by checking cache hit with hash key)
  cache.flush();
  const testText = 'cloud architect kubernetes aws';
  const testHash = cache.contentHash(testText);
  await cache.getOrEmbed(testText, async () => v3, 'hot');
  ok(vecEqual(cache.get(testHash), v3),                  'getOrEmbed: stored under contentHash key');

};
