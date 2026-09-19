'use strict';
/**
 * semantic-v2.test.js — Tests for new semantic.js v2 batch APIs.
 *
 * Tests:
 *   cosineAll()     — batch cosine computation
 *   rankAll()       — top-K ranking
 *   cosineBatch()   — multi-pair computation
 *   normalizeAll()  — batch normalisation
 */

const { ok, equal, near } = global._assert;

const { cosine, cosineAll, rankAll, cosineBatch, normalize, normalizeAll, composite } = require('../src/semantic');

function unit(values) {
  const mag = Math.sqrt(values.reduce((s, v) => s + v * v, 0));
  return new Float32Array(values.map(v => v / mag));
}

module.exports = function () {

  // ── cosineAll ──────────────────────────────────────────────────────────────

  const q  = unit([1, 0, 0]);
  const c1 = unit([1, 0, 0]);   // identical → 1.0
  const c2 = unit([0, 1, 0]);   // orthogonal → 0.0
  const c3 = unit([-1, 0, 0]);  // opposite → -1.0

  const scores = cosineAll(q, [c1, c2, c3]);
  ok(scores instanceof Float64Array,                       'cosineAll: returns Float64Array');
  equal(scores.length, 3,                                  'cosineAll: length matches corpus size');
  near(scores[0],  1.0, 0.001,                             'cosineAll: identical → 1.0');
  near(scores[1],  0.0, 0.001,                             'cosineAll: orthogonal → 0.0');
  near(scores[2], -1.0, 0.001,                             'cosineAll: opposite → -1.0');

  // Null / empty inputs
  const emptyScores = cosineAll(null, [c1, c2]);
  equal(emptyScores.length, 2,                             'cosineAll: null query → all 0');
  ok(emptyScores[0] === 0 && emptyScores[1] === 0,         'cosineAll: null query → zeroes');

  const noCorpus = cosineAll(q, []);
  equal(noCorpus.length, 0,                                'cosineAll: empty corpus → empty result');

  // Mismatched dimension → 0 for that entry
  const shortVec = unit([1, 0]);
  const mixed = cosineAll(q, [c1, shortVec, c3]);
  near(mixed[0],  1.0,  0.001,                             'cosineAll: normal entry computed');
  equal(mixed[1], 0,                                       'cosineAll: mismatched dim → 0');
  near(mixed[2], -1.0,  0.001,                             'cosineAll: entry after mismatch computed');

  // ── rankAll ───────────────────────────────────────────────────────────────

  const ranked = rankAll(q, [c3, c2, c1]);  // reverse order corpus
  ok(Array.isArray(ranked),                                'rankAll: returns Array');
  equal(ranked.length, 3,                                  'rankAll: returns all entries');

  // Sorted descending by score
  ok(ranked[0].score >= ranked[1].score,                   'rankAll: sorted descending [0]≥[1]');
  ok(ranked[1].score >= ranked[2].score,                   'rankAll: sorted descending [1]≥[2]');

  // Best match (c1, index 2 in input) should be first
  equal(ranked[0].index, 2,                                'rankAll: c1 (idx 2) is top result');
  near(ranked[0].score, 1.0, 0.001,                        'rankAll: top score ≈ 1.0');

  // topK limits results
  const top2 = rankAll(q, [c3, c2, c1], 2);
  equal(top2.length, 2,                                    'rankAll: topK=2 returns 2 results');

  const top1 = rankAll(q, [c3, c2, c1], 1);
  equal(top1.length, 1,                                    'rankAll: topK=1 returns 1 result');
  equal(top1[0].index, 2,                                  'rankAll: topK=1 returns best match');

  // topK larger than corpus → returns all
  const topBig = rankAll(q, [c1, c2], 99);
  equal(topBig.length, 2,                                  'rankAll: topK > corpus returns all');

  // ── cosineBatch ───────────────────────────────────────────────────────────

  const pairs = [[q, c1], [q, c2], [q, c3]];
  const batchScores = cosineBatch(pairs);
  ok(Array.isArray(batchScores),                           'cosineBatch: returns Array');
  equal(batchScores.length, 3,                             'cosineBatch: length matches pairs');
  near(batchScores[0],  1.0, 0.001,                        'cosineBatch: pair 0 ≈ 1.0');
  near(batchScores[1],  0.0, 0.001,                        'cosineBatch: pair 1 ≈ 0.0');
  near(batchScores[2], -1.0, 0.001,                        'cosineBatch: pair 2 ≈ -1.0');

  // Empty pairs → empty result
  equal(cosineBatch([]).length, 0,                         'cosineBatch: empty pairs → []');

  // ── normalizeAll ──────────────────────────────────────────────────────────

  const rawScores    = new Float64Array([0.90, 0.70, 0.50, -1.0]);
  const normScores   = normalizeAll(rawScores);

  ok(normScores instanceof Int32Array,                     'normalizeAll: returns Int32Array');
  equal(normScores.length, 4,                              'normalizeAll: length matches input');

  // High cosine → high normalized score
  ok(normScores[0] > 70,                                   'normalizeAll: cosine 0.90 → > 70');

  // Centre (0.70) → ~50
  near(normScores[1], 50, 3,                               'normalizeAll: cosine 0.70 → ~50');

  // Low cosine → low score
  ok(normScores[2] < 20,                                   'normalizeAll: cosine 0.50 → < 20');

  // cosine -1 → 0 (clamped)
  ok(normScores[3] >= 0,                                   'normalizeAll: cosine -1.0 → ≥ 0');

  // All values 0-100
  for (const s of normScores) {
    ok(s >= 0 && s <= 100,                                 `normalizeAll: score ${s} in [0,100]`);
  }

  // Custom opts propagate
  const highCentre = normalizeAll(new Float64Array([0.90]), { centre: 0.90 });
  near(highCentre[0], 50, 3,                               'normalizeAll: custom centre 0.90 → ~50');

  // ── cosine unrolled vs original: values match ─────────────────────────────
  // Generate a 384-dim vector and verify cosine result matches direct computation
  const dim = 384;
  const va = new Float32Array(dim).map((_, i) => Math.sin(i));
  const vb = new Float32Array(dim).map((_, i) => Math.cos(i));

  // Normalise
  const magA = Math.sqrt([...va].reduce((s, v) => s + v*v, 0));
  const magB = Math.sqrt([...vb].reduce((s, v) => s + v*v, 0));
  const vaN  = new Float32Array([...va].map(v => v / magA));
  const vbN  = new Float32Array([...vb].map(v => v / magB));

  const unrolled = cosine(vaN, vbN);
  // Direct non-unrolled computation
  let dotRef = 0;
  for (let i = 0; i < dim; i++) dotRef += vaN[i] * vbN[i];
  dotRef = Math.max(-1, Math.min(1, dotRef));

  near(unrolled, dotRef, 1e-5,                             'cosine unrolled: matches direct computation');

};
