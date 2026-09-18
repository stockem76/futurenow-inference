'use strict';
/**
 * semantic.js — Semantic scoring utilities for @futurenow/inference.
 *
 * Pure math — no I/O, no SupplyMatch imports, no dependencies.
 *
 * Exports:
 *   cosine(vecA, vecB)              → number -1..1
 *   normalize(cosine, opts?)        → number 0-100 (sigmoid)
 *   composite(smScore, kwScore, α, β) → number 0-100
 */

'use strict';

/**
 * Compute the dot-product cosine similarity between two unit-normalised
 * Float32Array vectors. Both must be the same length and L2-normalised.
 * Returns 0 for null/mismatched inputs rather than throwing.
 *
 * @param {Float32Array|number[]} a
 * @param {Float32Array|number[]} b
 * @returns {number} cosine similarity in [-1, 1]
 */
function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return Math.max(-1, Math.min(1, dot));
}

/**
 * Sigmoid normalisation — maps raw cosine [-1,1] to a [0,1] score.
 * Default parameters mirror SupplyMatch SM Score spec §O8:
 *   centre = 0.70  (cosine at which output = 0.50)
 *   scale  = 10    (steepness of the sigmoid)
 * These can be overridden via opts to tune for specific applications.
 *
 * Characteristic values (default):
 *   cosine 0.50 → ~0.12  (low match)
 *   cosine 0.70 → ~0.50  (neutral)
 *   cosine 0.90 → ~0.88  (strong match)
 *
 * @param {number} cos       — raw cosine value in [-1, 1]
 * @param {{ centre?: number, scale?: number, asPercent?: boolean }} [opts]
 * @returns {number} normalised score; 0–100 (integer) when asPercent is true (default),
 *                   0–1 float when asPercent is false
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
 * Blend a semantic score and a keyword score into a single composite score.
 * Mirrors SupplyMatch's demand match composite formula.
 *
 * @param {number} smScore  — semantic score [0–100]
 * @param {number} kwScore  — keyword score  [0–100]
 * @param {number} [α=0.40] — semantic weight (must sum to 1 with β)
 * @param {number} [β=0.60] — keyword weight
 * @returns {number} composite integer [0–100]
 */
function composite(smScore, kwScore, α = 0.40, β = 0.60) {
  return Math.round(Math.max(0, Math.min(100,
    α * (smScore || 0) + β * (kwScore || 0)
  )));
}

module.exports = { cosine, normalize, composite };
