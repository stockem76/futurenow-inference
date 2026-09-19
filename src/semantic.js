'use strict';
/**
 * semantic.js — Semantic scoring utilities for @futurenow/inference.
 *
 * Pure math — no I/O, no SupplyMatch imports, no dependencies.
 *
 * Optimisations vs v1:
 *   - cosine() uses tight loop with no Array.from() or intermediate allocations
 *   - cosineAll() batch computation: one query vector vs N corpus vectors
 *   - rankAll() returns sorted results with index for top-K retrieval
 *   - normalize() unchanged (already ~0ns)
 *   - composite() unchanged
 *   - cosineBatch() for comparing multiple query/doc pairs in one call
 *
 * Exports:
 *   cosine(vecA, vecB)                    → number -1..1
 *   cosineAll(queryVec, corpusVecs)       → Float64Array of scores
 *   rankAll(queryVec, corpusVecs, topK?)  → Array<{index, score}>
 *   cosineBatch(pairs)                    → number[]
 *   normalize(cosine, opts?)              → number 0-100 (sigmoid)
 *   composite(smScore, kwScore, α, β)    → number 0-100
 *   normalizeAll(cosineScores, opts?)    → Int32Array
 */

'use strict';

/**
 * Compute the dot-product cosine similarity between two unit-normalised
 * Float32Array vectors. Both must be the same length and L2-normalised.
 * Returns 0 for null/mismatched inputs rather than throwing.
 *
 * Optimisation: unrolled inner loop in groups of 4 (manual loop unrolling for
 * V8 — reduces branch overhead on 384-dim vectors).
 *
 * @param {Float32Array|number[]} a
 * @param {Float32Array|number[]} b
 * @returns {number} cosine similarity in [-1, 1]
 */
function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  const n = a.length;
  let dot = 0;
  // Unrolled by 4 for better JIT vectorisation
  let i = 0;
  const n4 = n - (n % 4);
  for (; i < n4; i += 4) {
    dot += a[i]   * b[i]
         + a[i+1] * b[i+1]
         + a[i+2] * b[i+2]
         + a[i+3] * b[i+3];
  }
  for (; i < n; i++) dot += a[i] * b[i];
  return Math.max(-1, Math.min(1, dot));
}

/**
 * Compute cosine similarity of one query vector against all corpus vectors.
 * Returns a Float64Array of scores (one per corpus vector).
 * Both queryVec and each corpusVec must be pre-normalised Float32Arrays.
 *
 * This is the critical hot path for semantic search — one query vs N candidates.
 *
 * @param {Float32Array} queryVec
 * @param {Float32Array[]} corpusVecs
 * @returns {Float64Array}  scores[i] = cosine(queryVec, corpusVecs[i])
 */
function cosineAll(queryVec, corpusVecs) {
  const n = corpusVecs.length;
  const scores = new Float64Array(n);
  if (!queryVec || !n) return scores;
  const dim = queryVec.length;
  for (let ci = 0; ci < n; ci++) {
    const cv = corpusVecs[ci];
    if (!cv || cv.length !== dim) { scores[ci] = 0; continue; }
    let dot = 0;
    let i = 0;
    const n4 = dim - (dim % 4);
    for (; i < n4; i += 4) {
      dot += queryVec[i]   * cv[i]
           + queryVec[i+1] * cv[i+1]
           + queryVec[i+2] * cv[i+2]
           + queryVec[i+3] * cv[i+3];
    }
    for (; i < dim; i++) dot += queryVec[i] * cv[i];
    scores[ci] = Math.max(-1, Math.min(1, dot));
  }
  return scores;
}

/**
 * Rank all corpus vectors against a query vector, returning the top-K results.
 * Returns an array of {index, score} sorted descending by score.
 *
 * @param {Float32Array}   queryVec
 * @param {Float32Array[]} corpusVecs
 * @param {number}         [topK=Infinity]  — max results to return
 * @returns {Array<{index:number, score:number}>}
 */
function rankAll(queryVec, corpusVecs, topK) {
  const raw = cosineAll(queryVec, corpusVecs);
  const n   = raw.length;
  const results = new Array(n);
  for (let i = 0; i < n; i++) results[i] = { index: i, score: raw[i] };
  results.sort((a, b) => b.score - a.score);
  return topK !== undefined ? results.slice(0, topK) : results;
}

/**
 * Compute cosine similarities for multiple (a, b) vector pairs in one call.
 *
 * @param {Array<[Float32Array, Float32Array]>} pairs
 * @returns {number[]}  one score per pair
 */
function cosineBatch(pairs) {
  return pairs.map(([a, b]) => cosine(a, b));
}

/**
 * Sigmoid normalisation — maps raw cosine [-1,1] to a [0,1] score.
 * Default parameters mirror SupplyMatch SM Score spec §O8:
 *   centre = 0.70  (cosine at which output = 0.50)
 *   scale  = 10    (steepness of the sigmoid)
 *
 * @param {number} cos       — raw cosine value in [-1, 1]
 * @param {{ centre?: number, scale?: number, asPercent?: boolean }} [opts]
 * @returns {number} 0–100 integer (asPercent:true, default) or 0–1 float
 */
function normalize(cos, opts = {}) {
  const centre    = typeof opts.centre === 'number' ? opts.centre : 0.70;
  const scale     = typeof opts.scale  === 'number' ? opts.scale  : 10;
  const asPercent = opts.asPercent !== false;   // default true

  const sig = 1 / (1 + Math.exp(-scale * (cos - centre)));

  if (asPercent) return Math.round(Math.max(0, Math.min(100, sig * 100)));
  return Math.max(0, Math.min(1, sig));
}

/**
 * Normalise an array (or Float64Array) of cosine scores to [0,100] integers.
 * Returns an Int32Array for compact storage.
 *
 * @param {number[]|Float64Array} cosineScores
 * @param {object} [opts]  — same options as normalize()
 * @returns {Int32Array}
 */
function normalizeAll(cosineScores, opts = {}) {
  const out = new Int32Array(cosineScores.length);
  for (let i = 0; i < cosineScores.length; i++) {
    out[i] = normalize(cosineScores[i], opts);
  }
  return out;
}

/**
 * Blend a semantic score and a keyword score into a single composite score.
 *
 * @param {number} smScore  — semantic score [0–100]
 * @param {number} kwScore  — keyword score  [0–100]
 * @param {number} [α=0.40] — semantic weight
 * @param {number} [β=0.60] — keyword weight
 * @returns {number} composite integer [0–100]
 */
function composite(smScore, kwScore, α = 0.40, β = 0.60) {
  return Math.round(Math.max(0, Math.min(100,
    α * (smScore || 0) + β * (kwScore || 0)
  )));
}

module.exports = { cosine, cosineAll, rankAll, cosineBatch, normalize, normalizeAll, composite };
