'use strict';
/**
 * keyword.test.js — unit tests for @futurenow/inference src/keyword.js
 *
 * Tests the exported API:
 *   score(practSkills, reqSkills, niceSkills, opts) → {score, tiers, matched, gaps}
 *   computeKeywordScore(a, b, weights?)             → {composite, breakdown}
 *   jaroWinkler(a, b)                               (not exported directly; tested via computeKeywordScore)
 *
 * Pure functions — no I/O, no init() required.
 */

const { ok, equal, near } = global._assert;

const keyword = require('../src/keyword');

module.exports = function () {

  // ── computeKeywordScore — string vs string ─────────────────────────────────

  const exact   = keyword.computeKeywordScore('react', 'react');
  const none    = keyword.computeKeywordScore('react', 'python');
  const partial = keyword.computeKeywordScore('react', 'reactjs');

  ok(exact.composite >= 90,                                'exact match → composite ≥ 90');
  ok(none.composite  < 30,                                 'no overlap  → composite < 30');
  ok(partial.composite > none.composite,                   'partial match > no overlap');
  ok(partial.composite < exact.composite,                  'partial match < exact');

  // Bounded 0–100
  ok(exact.composite   >= 0 && exact.composite   <= 100,  'exact  bounded 0–100');
  ok(none.composite    >= 0 && none.composite    <= 100,  'none   bounded 0–100');
  ok(partial.composite >= 0 && partial.composite <= 100,  'partial bounded 0–100');

  // breakdown structure
  ok(Array.isArray(exact.breakdown),                       'breakdown is an array');
  ok(exact.breakdown.length > 0,                           'breakdown has entries');
  const b0 = exact.breakdown[0];
  ok(typeof b0.algorithm    === 'string',                  'breakdown[0].algorithm is string');
  ok(typeof b0.rawScore     === 'number',                  'breakdown[0].rawScore is number');
  ok(typeof b0.weight       === 'number',                  'breakdown[0].weight is number');
  ok(typeof b0.contribution === 'number',                  'breakdown[0].contribution is number');

  // ── score — skill-set scoring ──────────────────────────────────────────────

  // Exact required skill match
  const r1 = keyword.score(['react'], ['react'], []);
  ok(typeof r1.score === 'number',                         'score() returns {score}');
  ok(r1.score >= 80,                                       'single exact required skill → score ≥ 80');
  ok(Array.isArray(r1.matched),                            'score() returns {matched}');
  ok(Array.isArray(r1.gaps),                               'score() returns {gaps}');
  ok(Array.isArray(r1.tiers),                              'score() returns {tiers}');

  // Missing required skill should appear in gaps
  const r2 = keyword.score(['javascript'], ['python'], []);
  ok(r2.gaps.includes('python') || r2.score < 50,         'missing required skill in gaps or low score');

  // Multiple skills: matching all required
  const r3 = keyword.score(
    ['react', 'typescript', 'node'],
    ['react', 'typescript'],
    ['node']
  );
  ok(r3.score > 70,                                        'matching all required+nice → score > 70');
  ok(r3.matched.length >= 2,                               'both required skills matched');

  // Empty practitioner — should be very low
  const r4 = keyword.score([], ['react', 'typescript'], []);
  ok(r4.score === 0,                                       'empty practitioner → score = 0');
  ok(r4.gaps.length === 2,                                 'empty practitioner → both required in gaps');

  // Empty required and nice — should return 0
  const r5 = keyword.score(['react'], [], []);
  equal(r5.score, 0,                                       'no required/nice skills → score = 0');

  // ── DEFAULT_WEIGHTS structure ──────────────────────────────────────────────

  const dw = keyword.DEFAULT_WEIGHTS;
  ok(typeof dw === 'object' && dw !== null,                'DEFAULT_WEIGHTS is an object');
  ok(typeof dw.jaroWinkler    === 'number',                'DEFAULT_WEIGHTS.jaroWinkler is number');
  ok(typeof dw.jaccard        === 'number',                'DEFAULT_WEIGHTS.jaccard is number');
  ok(typeof dw.levenshtein    === 'number',                'DEFAULT_WEIGHTS.levenshtein is number');

  // Weights should sum to ~1
  const wSum = Object.values(dw).reduce((s, v) => s + v, 0);
  near(wSum, 1.0, 0.001,                                   'DEFAULT_WEIGHTS sum ≈ 1.0');

};
