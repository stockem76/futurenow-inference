'use strict';
/**
 * baseline-profile.js — End-to-end latency and throughput profiling
 * for @futurenow/inference keyword, semantic, and AI match pipelines.
 *
 * Run: node bench/baseline-profile.js [--iters N]
 * Writes results to bench/baseline-results.json
 */

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

const keyword  = require('../src/keyword');
const semantic = require('../src/semantic');

// ── Fixtures ──────────────────────────────────────────────────────────────────

const PRACTITIONER_SKILLS = [
  'React', 'TypeScript', 'Node.js', 'AWS', 'Docker', 'Kubernetes',
  'GraphQL', 'PostgreSQL', 'Redis', 'Python', 'TensorFlow', 'CI/CD',
];

const REQ_SKILLS  = ['React', 'TypeScript', 'AWS', 'Docker'];
const NICE_SKILLS = ['Kubernetes', 'GraphQL', 'Redis'];

const PHRASES = [
  'machine learning engineer',
  'senior software developer',
  'cloud architect',
  'data scientist',
  'devops engineer',
  'full stack developer',
  'platform engineer',
  'site reliability engineer',
];

// Unit vectors for cosine benchmarking
function makeUnitVec(dim, seed) {
  const v = new Float32Array(dim);
  let s = seed || 1;
  for (let i = 0; i < dim; i++) {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    v[i] = ((s >>> 0) / 0xffffffff) * 2 - 1;
  }
  let mag = 0;
  for (let i = 0; i < dim; i++) mag += v[i] * v[i];
  mag = Math.sqrt(mag);
  for (let i = 0; i < dim; i++) v[i] /= mag;
  return v;
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

function bench(label, fn, iters) {
  // Warm up
  for (let i = 0; i < Math.min(10, iters); i++) fn(i);

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  const elapsed = performance.now() - t0;

  return {
    label,
    iters,
    totalMs: +elapsed.toFixed(3),
    avgMs:   +(elapsed / iters).toFixed(4),
    opsPerSec: Math.round(iters / (elapsed / 1000)),
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────

const args  = process.argv.slice(2);
const iters = parseInt(args[args.indexOf('--iters') + 1] || '10000');

console.log(`\n@futurenow/inference — Baseline Profile (${iters} iters each)\n`);

const results = [];

// 1. computeKeywordScore — string pair
results.push(bench('keyword.computeKeywordScore (exact)', i => {
  keyword.computeKeywordScore('React', 'React');
}, iters));

results.push(bench('keyword.computeKeywordScore (partial)', i => {
  const a = PHRASES[i % PHRASES.length];
  const b = PHRASES[(i + 3) % PHRASES.length];
  keyword.computeKeywordScore(a, b);
}, iters));

// 2. keyword.score — skill-set scoring
results.push(bench('keyword.score (12 pract skills vs 4+3)', i => {
  keyword.score(PRACTITIONER_SKILLS, REQ_SKILLS, NICE_SKILLS);
}, iters));

// 3. semantic.cosine — 384-dim unit vectors
const vecA = makeUnitVec(384, 42);
const vecB = makeUnitVec(384, 99);
results.push(bench('semantic.cosine (384-dim Float32Array)', i => {
  semantic.cosine(vecA, vecB);
}, iters));

// 4. semantic.normalize
results.push(bench('semantic.normalize', i => {
  semantic.normalize(0.5 + (i % 100) / 200);
}, iters));

// 5. semantic.composite
results.push(bench('semantic.composite', i => {
  semantic.composite(60 + (i % 40), 70 + (i % 30));
}, iters));

// 6. BM25 simulation (keyword score with IDF weighting — baseline for comparison)
// Simulates cost of repeated keyword.score across 50 candidates
results.push(bench('keyword.score × 50 candidates (batch)', i => {
  const candidates = Array.from({ length: 50 }, (_, j) =>
    [`skill${j}`, `tech${j}`, `lang${j}`]
  );
  for (const c of candidates) {
    keyword.score(c, REQ_SKILLS, NICE_SKILLS);
  }
}, Math.floor(iters / 10)));

// 7. Cosine × 50 vectors (batch semantic search simulation)
const vecs = Array.from({ length: 50 }, (_, i) => makeUnitVec(384, i));
results.push(bench('semantic.cosine × 50 vectors (batch)', i => {
  for (const v of vecs) semantic.cosine(vecA, v);
}, Math.floor(iters / 10)));

// ── Report ────────────────────────────────────────────────────────────────────

const colW = [50, 10, 12, 12, 14];
function padR(s, n) { return String(s).padEnd(n); }
function padL(s, n) { return String(s).padStart(n); }

console.log(
  padR('Benchmark', colW[0]) +
  padL('Iters',    colW[1]) +
  padL('Total ms', colW[2]) +
  padL('Avg ms',   colW[3]) +
  padL('ops/sec',  colW[4])
);
console.log('─'.repeat(colW.reduce((a,b) => a+b, 0)));

for (const r of results) {
  console.log(
    padR(r.label, colW[0]) +
    padL(r.iters,      colW[1]) +
    padL(r.totalMs,    colW[2]) +
    padL(r.avgMs,      colW[3]) +
    padL(r.opsPerSec,  colW[4])
  );
}

const outPath = path.join(__dirname, 'baseline-results.json');
fs.writeFileSync(outPath, JSON.stringify({ timestamp: new Date().toISOString(), iters, results }, null, 2));
console.log(`\nResults written to: ${outPath}\n`);
