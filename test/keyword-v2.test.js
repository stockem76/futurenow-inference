'use strict';
/**
 * keyword-v2.test.js — Tests for new keyword.js v2 algorithms.
 *
 * Tests:
 *   bm25Score()          — BM25+ single-doc scoring
 *   tfidfPositional()    — positional TF-IDF
 *   buildInvertedIndex() — inverted index construction
 *   proximityScore()     — proximity scoring via index
 *   DEFAULT_WEIGHTS      — includes bm25 and tfidfPositional
 *   score() fast-path    — empty practitioner, early exit
 */

const { ok, equal, near } = global._assert;

const keyword = require('../src/keyword');

module.exports = function () {

  // ── bm25Score ──────────────────────────────────────────────────────────────

  // Exact single-term match should score > 0
  const bm25Exact = keyword.bm25Score('react', 'react developer');
  ok(bm25Exact > 0,                                       'bm25: exact term match → score > 0');
  ok(bm25Exact <= 1,                                      'bm25: score bounded ≤ 1');

  // Query term not in document → score 0
  const bm25Miss = keyword.bm25Score('python', 'react developer');
  equal(bm25Miss, 0,                                      'bm25: no matching term → score 0');

  // Multi-term match should score higher than single-term
  const bm25Multi = keyword.bm25Score('react typescript', 'react typescript developer');
  ok(bm25Multi > bm25Exact,                               'bm25: multi-term match > single-term');
  ok(bm25Multi <= 1,                                      'bm25: multi-term score bounded ≤ 1');

  // Empty query or doc → 0
  equal(keyword.bm25Score('',       'react'), 0,          'bm25: empty query → 0');
  equal(keyword.bm25Score('react',  ''),      0,          'bm25: empty doc → 0');
  equal(keyword.bm25Score('',       ''),      0,          'bm25: both empty → 0');

  // Symmetric: bidirectional scoring should both be > 0
  const bm25ab = keyword.bm25Score('kubernetes', 'kubernetes orchestration');
  const bm25ba = keyword.bm25Score('kubernetes orchestration', 'kubernetes');
  ok(bm25ab > 0 && bm25ba > 0,                            'bm25: bidirectional scoring both > 0');

  // ── tfidfPositional ────────────────────────────────────────────────────────

  // First-position match should score higher than middle-position
  const tfidfFirst  = keyword.tfidfPositional('react', 'react developer engineer');
  const tfidfMiddle = keyword.tfidfPositional('react', 'senior react developer');
  ok(tfidfFirst > tfidfMiddle,                            'tfidf: first-position match > middle');

  // Exact match → score > 0
  const tfidfExact = keyword.tfidfPositional('react', 'react');
  ok(tfidfExact > 0,                                      'tfidf: exact match → score > 0');
  ok(tfidfExact <= 1,                                     'tfidf: score bounded ≤ 1');

  // No match → 0
  equal(keyword.tfidfPositional('python', 'react developer'), 0, 'tfidf: no match → 0');

  // Empty inputs → 0
  equal(keyword.tfidfPositional('', 'react'),  0,         'tfidf: empty query → 0');
  equal(keyword.tfidfPositional('react', ''),  0,         'tfidf: empty doc → 0');

  // ── buildInvertedIndex ────────────────────────────────────────────────────

  const docs  = ['react developer', 'typescript engineer', 'react typescript senior'];
  const index = keyword.buildInvertedIndex(docs);

  ok(index instanceof Map,                                'buildInvertedIndex: returns Map');
  ok(index.has('react'),                                  'buildInvertedIndex: react term indexed');
  ok(index.has('typescript'),                             'buildInvertedIndex: typescript indexed');
  ok(index.has('developer'),                              'buildInvertedIndex: developer indexed');

  // 'react' appears in doc 0 and doc 2
  const reactEntry = index.get('react');
  const docIdxs    = reactEntry.map(e => e.docIdx);
  ok(docIdxs.includes(0),                                'react appears in doc 0');
  ok(docIdxs.includes(2),                                'react appears in doc 2');

  // Position check: 'react' is at position 0 in doc 0
  const reactInDoc0 = reactEntry.find(e => e.docIdx === 0);
  ok(reactInDoc0 && reactInDoc0.positions.includes(0),   'react at position 0 in doc 0');

  // Empty docs array
  const emptyIdx = keyword.buildInvertedIndex([]);
  ok(emptyIdx instanceof Map && emptyIdx.size === 0,      'empty docs → empty index');

  // ── proximityScore ────────────────────────────────────────────────────────

  // Both terms in index and close → high score
  const psIndex = keyword.buildInvertedIndex(['react typescript developer']);
  const psHigh  = keyword.proximityScore(['react', 'typescript'], psIndex);
  ok(psHigh > 0.5,                                        'proximityScore: close terms → score > 0.5');
  ok(psHigh <= 1,                                         'proximityScore: score bounded ≤ 1');

  // One term in index, one not → lower score
  const psPartial = keyword.proximityScore(['react', 'python'], psIndex);
  ok(psPartial < psHigh,                                  'proximityScore: one miss < both present');
  ok(psPartial > 0,                                       'proximityScore: one hit → score > 0');

  // Neither term → 0
  const psNone = keyword.proximityScore(['java', 'scala'], psIndex);
  equal(psNone, 0,                                        'proximityScore: no terms in index → 0');

  // Empty query → 0
  equal(keyword.proximityScore([], psIndex), 0,           'proximityScore: empty terms → 0');

  // ── DEFAULT_WEIGHTS now includes bm25 + tfidfPositional ──────────────────

  const dw = keyword.DEFAULT_WEIGHTS;
  ok(typeof dw.bm25            === 'number',              'DEFAULT_WEIGHTS.bm25 is number');
  ok(typeof dw.tfidfPositional === 'number',              'DEFAULT_WEIGHTS.tfidfPositional is number');
  ok(dw.bm25 > 0,                                         'DEFAULT_WEIGHTS.bm25 > 0');
  ok(dw.tfidfPositional > 0,                              'DEFAULT_WEIGHTS.tfidfPositional > 0');

  // All weights sum to ~1
  const wSum = Object.values(dw).reduce((s, v) => s + v, 0);
  near(wSum, 1.0, 0.001,                                  'DEFAULT_WEIGHTS (v2) sum ≈ 1.0');

  // ── ALGO_META includes new algorithms ─────────────────────────────────────

  const meta = keyword.ALGO_META;
  ok(meta.bm25            !== undefined,                  'ALGO_META.bm25 present');
  ok(meta.tfidfPositional !== undefined,                  'ALGO_META.tfidfPositional present');
  ok(typeof meta.bm25.label === 'string',                 'ALGO_META.bm25.label is string');
  ok(typeof meta.tfidfPositional.useCase === 'string',    'ALGO_META.tfidfPositional.useCase is string');

  // ── computeKeywordScore breakdown includes new algorithms ─────────────────

  const cs = keyword.computeKeywordScore('react', 'react developer');
  const algos = cs.breakdown.map(b => b.algorithm);
  ok(algos.includes('bm25'),                              'breakdown includes bm25');
  ok(algos.includes('tfidfPositional'),                   'breakdown includes tfidfPositional');

  // ── Empty practitioner fast-path ──────────────────────────────────────────

  const emptyPractResult = keyword.score([], ['react', 'typescript'], ['node']);
  equal(emptyPractResult.score, 0,                        'empty pract fast-path: score = 0');
  equal(emptyPractResult.gaps.length, 2,                  'empty pract fast-path: 2 required gaps');
  ok(Array.isArray(emptyPractResult.matched),             'empty pract fast-path: matched array');
  equal(emptyPractResult.matched.length, 0,               'empty pract fast-path: no matches');
  // tiers still returned for all skills (req + nice)
  equal(emptyPractResult.tiers.length, 3,                 'empty pract fast-path: tiers for all skills');

};
