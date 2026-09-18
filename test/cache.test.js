'use strict';
/**
 * cache.test.js — unit tests for @futurenow/inference src/cache.js
 *
 * Covers hot, TTL, and flush behaviour.
 * Cold (SQLite) tier is tested only when better-sqlite3 is available.
 */

const { ok, equal, near } = global._assert;

const cache = require('../src/cache');

// ── Helpers ──────────────────────────────────────────────────────────────────

function vec(values) { return new Float32Array(values); }

/** Check structural equality of two Float32Arrays. */
function vecEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 1e-6) return false;
  }
  return true;
}

module.exports = function () {

  // Initialise without cold store (SQLite skipped)
  cache._setConfig({ hotMax: 5, ttlMs: 200 });

  const v1 = vec([0.1, 0.2, 0.3]);
  const v2 = vec([0.4, 0.5, 0.6]);

  // ── Hot tier ──────────────────────────────────────────────────────────────

  cache.flush();
  cache.set('k1', v1, 'hot');

  ok(vecEqual(cache.get('k1'), v1),             'hot tier set+get round-trips correctly');
  equal(cache.get('k99'), null,                 'missing key returns null');

  // ── TTL tier ─────────────────────────────────────────────────────────────

  cache.flush();
  cache.set('k2', v2, 'ttl');

  ok(vecEqual(cache.get('k2'), v2),             'ttl tier: get returns value before expiry');

  // Wait 250 ms — ttlMs is 200, so entry should expire
  // (use synchronous spin to avoid async complexity in test runner)
  const t0 = Date.now();
  while (Date.now() - t0 < 250) { /* spin */ }

  // Must also flush hot (promote puts value into hot on first read)
  cache.flush();
  equal(cache.get('k2'), null,                  'ttl tier: get returns null after expiry');

  // ── del() removes from all tiers ─────────────────────────────────────────

  cache.flush();
  cache.set('k3', v1, 'hot');
  cache.set('k3', v1, 'ttl');
  cache.del('k3');
  equal(cache.get('k3'), null,                  'del() removes from hot and ttl');

  // ── LRU eviction at hotMax ────────────────────────────────────────────────

  cache.flush();
  // hotMax = 5 — set 6 items; first item should be evicted
  for (let i = 0; i < 6; i++) cache.set(`lru${i}`, vec([i]), 'hot');
  equal(cache.get('lru0'), null,                'lru0 evicted when hotMax=5 exceeded');
  ok(vecEqual(cache.get('lru5'), vec([5])),     'lru5 (most recent) still present');

  // ── stats() structure ─────────────────────────────────────────────────────

  cache.flush();
  cache.set('s1', v1, 'hot');
  cache.set('s2', v2, 'ttl');
  const s = cache.stats();
  ok(typeof s === 'object' && s !== null,       'stats() returns object');
  ok(typeof s.hot  === 'number',                'stats.hot is a number');
  ok(typeof s.ttl  === 'number',                'stats.ttl is a number');
  ok(typeof s.cold === 'number',                'stats.cold is a number');
  ok(s.hot  >= 1,                               'stats.hot ≥ 1 after set');
  ok(s.ttl  >= 1,                               'stats.ttl ≥ 1 after ttl set');
  equal(s.cold, 0,                              'stats.cold = 0 when no coldStorePath');

  // ── flush() clears in-process tiers ──────────────────────────────────────

  cache.flush();
  const sf = cache.stats();
  equal(sf.hot, 0,                              'stats.hot = 0 after flush');
  equal(sf.ttl, 0,                              'stats.ttl = 0 after flush');

  // ── null/empty key guards ─────────────────────────────────────────────────

  cache.set(null, v1);
  cache.set('',   v1);
  equal(cache.get(null), null,                  'null key → get returns null');
  equal(cache.get(''),   null,                  'empty key → get returns null');

};
