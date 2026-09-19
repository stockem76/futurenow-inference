'use strict';
/**
 * cache.js — 3-tier vector cache for @futurenow/inference.
 *
 * Tier 1 — Hot Map:  in-process Map<key, Float32Array>, instant lookup, bounded
 *                    by maxHot entries (LRU-style eviction).
 * Tier 2 — TTL Map:  Map<key, {vec, expiresAt}> for medium-duration items,
 *                    cleared on reads past TTL.
 * Tier 3 — SQLite:   optional persistent cold store; only opened when
 *                    opts.coldStorePath is provided.
 *
 * v2 improvements:
 *   - getMany(keys) / setMany(entries) — batch read/write for pipeline throughput
 *   - contentHash(text) — deterministic cache key from text content
 *   - getOrEmbed(text, embedFn) — cache-aware single-text embed helper
 *   - LRU eviction improved: Map preserves insertion order, O(1) oldest key
 *
 * No external runtime dependencies. SQLite is accessed via the `better-sqlite3`
 * optional peer dependency; if it is not installed the cold tier is silently
 * skipped.
 *
 * Exports:
 *   get(key)                         → Float32Array | null
 *   getMany(keys)                    → Map<string, Float32Array|null>
 *   set(key, vec, tier?)             → void
 *   setMany(entries, tier?)          → void
 *   del(key)                         → void
 *   flush()                          → void  (all tiers)
 *   stats()                          → {hot, ttl, cold}
 *   contentHash(text)                → string  (deterministic key)
 *   getOrEmbed(text, embedFn, tier?) → Promise<Float32Array|null>
 *   _setConfig(cfg)                  → called by index.js after init()
 */

let _config = null;
let _db     = null;   // better-sqlite3 handle (optional)
let _dbReady = false;

function _setConfig(cfg) {
  _config = cfg;
  _dbReady = false;
  _db      = null;
  _hotMap.clear();
  _ttlMap.clear();
  _hotOrder.length = 0;
}

// ── Constants / defaults ──────────────────────────────────────────────────────

const DEFAULT_MAX_HOT  = 500;          // maximum hot-tier entries
const DEFAULT_TTL_MS   = 10 * 60_000;  // 10 minutes for TTL tier

// ── Tier 1 — Hot Map (LRU-like, bounded) ─────────────────────────────────────

/** @type {Map<string, Float32Array>} */
const _hotMap   = new Map();
/** @type {string[]}  insertion-order key list for eviction */
const _hotOrder = [];

function _hotMaxSize() {
  return (_config && _config.hotMax) ? _config.hotMax : DEFAULT_MAX_HOT;
}

function _hotGet(key) {
  return _hotMap.get(key) ?? null;
}

function _hotSet(key, vec) {
  if (_hotMap.has(key)) {
    _hotMap.set(key, vec);
    return;
  }
  const max = _hotMaxSize();
  if (_hotOrder.length >= max) {
    const oldest = _hotOrder.shift();
    if (oldest) _hotMap.delete(oldest);
  }
  _hotMap.set(key, vec);
  _hotOrder.push(key);
}

function _hotDel(key) {
  if (!_hotMap.has(key)) return;
  _hotMap.delete(key);
  const idx = _hotOrder.indexOf(key);
  if (idx !== -1) _hotOrder.splice(idx, 1);
}

// ── Tier 2 — TTL Map ──────────────────────────────────────────────────────────

/** @type {Map<string, {vec: Float32Array, expiresAt: number}>} */
const _ttlMap = new Map();

function _ttlMs() {
  return (_config && _config.ttlMs) ? _config.ttlMs : DEFAULT_TTL_MS;
}

function _ttlGet(key) {
  const entry = _ttlMap.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    _ttlMap.delete(key);
    return null;
  }
  return entry.vec;
}

function _ttlSet(key, vec) {
  _ttlMap.set(key, { vec, expiresAt: Date.now() + _ttlMs() });
}

function _ttlDel(key) {
  _ttlMap.delete(key);
}

// ── Tier 3 — SQLite cold store ────────────────────────────────────────────────

function _ensureDb() {
  if (_dbReady) return !!_db;
  _dbReady = true;

  const coldPath = _config && _config.coldStorePath;
  if (!coldPath) return false;

  try {
    const Database = require('better-sqlite3');   // eslint-disable-line global-require
    _db = new Database(coldPath);
    _db.pragma('journal_mode = WAL');
    _db.exec(`
      CREATE TABLE IF NOT EXISTS vector_cache (
        cache_key   TEXT PRIMARY KEY,
        vector_data BLOB NOT NULL,
        created_at  INTEGER NOT NULL
      );
    `);
    return true;
  } catch (_err) {
    _db = null;
    return false;
  }
}

function _vecToBlob(vec) {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

function _blobToVec(blob) {
  const ab = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
  return new Float32Array(ab);
}

function _coldGet(key) {
  if (!_ensureDb() || !_db) return null;
  try {
    const row = _db.prepare('SELECT vector_data FROM vector_cache WHERE cache_key = ?').get(key);
    return row ? _blobToVec(row.vector_data) : null;
  } catch (_) {
    return null;
  }
}

function _coldSet(key, vec) {
  if (!_ensureDb() || !_db) return;
  try {
    _db.prepare(`
      INSERT INTO vector_cache (cache_key, vector_data, created_at)
      VALUES (?, ?, ?)
      ON CONFLICT(cache_key) DO UPDATE SET vector_data = excluded.vector_data,
                                            created_at  = excluded.created_at
    `).run(key, _vecToBlob(vec), Date.now());
  } catch (_) { /* ignore write failures */ }
}

function _coldDel(key) {
  if (!_ensureDb() || !_db) return;
  try {
    _db.prepare('DELETE FROM vector_cache WHERE cache_key = ?').run(key);
  } catch (_) { /* ignore */ }
}

function _coldCount() {
  if (!_ensureDb() || !_db) return 0;
  try {
    return _db.prepare('SELECT COUNT(*) AS n FROM vector_cache').get().n;
  } catch (_) { return 0; }
}

// ── Content hashing ───────────────────────────────────────────────────────────

/**
 * Produce a deterministic cache key from text content.
 * Uses a fast non-cryptographic hash (djb2a variant) suitable for cache keys.
 * Produces a hex string of 8 chars.
 *
 * @param {string} text
 * @returns {string}
 */
function contentHash(text) {
  const s = String(text || '');
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h) ^ s.charCodeAt(i);
    h = h >>> 0;  // keep as unsigned 32-bit
  }
  return h.toString(16).padStart(8, '0');
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Look up a vector by key. Checks hot → ttl → cold in order.
 * Promotes cold hits into the hot tier automatically.
 *
 * @param {string} key
 * @returns {Float32Array|null}
 */
function get(key) {
  let vec = _hotGet(key);
  if (vec) return vec;

  vec = _ttlGet(key);
  if (vec) {
    _hotSet(key, vec);
    return vec;
  }

  vec = _coldGet(key);
  if (vec) {
    _hotSet(key, vec);
    return vec;
  }

  return null;
}

/**
 * Batch read — returns a Map of key → Float32Array|null.
 * Hits are promoted to hot tier automatically.
 *
 * @param {string[]} keys
 * @returns {Map<string, Float32Array|null>}
 */
function getMany(keys) {
  const result = new Map();
  for (const key of keys) {
    result.set(key, get(key));
  }
  return result;
}

/**
 * Store a vector in the cache.
 *
 * @param {string}       key
 * @param {Float32Array} vec
 * @param {'hot'|'ttl'|'cold'} [tier='hot']
 */
function set(key, vec, tier = 'hot') {
  if (!key || !vec) return;
  if (tier === 'cold') {
    _hotSet(key, vec);
    _ttlSet(key, vec);
    _coldSet(key, vec);
  } else if (tier === 'ttl') {
    _hotSet(key, vec);
    _ttlSet(key, vec);
  } else {
    _hotSet(key, vec);
  }
}

/**
 * Batch write — stores multiple key/vec pairs in one call.
 *
 * @param {Array<{key:string, vec:Float32Array}>} entries
 * @param {'hot'|'ttl'|'cold'} [tier='hot']
 */
function setMany(entries, tier = 'hot') {
  for (const { key, vec } of entries) {
    set(key, vec, tier);
  }
}

/**
 * Remove a key from all tiers.
 * @param {string} key
 */
function del(key) {
  _hotDel(key);
  _ttlDel(key);
  _coldDel(key);
}

/**
 * Clear all in-process tiers. Does NOT wipe the SQLite cold store.
 */
function flush() {
  _hotMap.clear();
  _hotOrder.length = 0;
  _ttlMap.clear();
}

/**
 * Return entry counts for each tier.
 * @returns {{ hot: number, ttl: number, cold: number }}
 */
function stats() {
  const now = Date.now();
  for (const [k, v] of _ttlMap) {
    if (now > v.expiresAt) _ttlMap.delete(k);
  }
  return {
    hot:  _hotMap.size,
    ttl:  _ttlMap.size,
    cold: _coldCount(),
  };
}

/**
 * Cache-aware single embedding helper.
 * Looks up the cache by contentHash(text); calls embedFn(text) on miss
 * and stores the result at the given tier.
 *
 * @param {string}   text
 * @param {function} embedFn         — async (text) → Float32Array|null
 * @param {'hot'|'ttl'|'cold'} [tier='ttl']
 * @returns {Promise<Float32Array|null>}
 */
async function getOrEmbed(text, embedFn, tier = 'ttl') {
  const key = contentHash(text);
  const cached = get(key);
  if (cached) return cached;

  const vec = await embedFn(text);
  if (vec) set(key, vec, tier);
  return vec || null;
}

module.exports = { get, getMany, set, setMany, del, flush, stats, contentHash, getOrEmbed, _setConfig };
