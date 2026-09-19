'use strict';
/**
 * experiment-harness.js — Randomised practitioner experiment loop.
 *
 * Implements a continuously randomised search over hyperparameters to find
 * the best-performing keyword weight configuration and match thresholds.
 *
 * Search strategy: Simulated annealing with random restarts.
 *   - Starts from multiple random seeds
 *   - Mutates the current best configuration with decreasing step size
 *   - Accepts worse configurations with probability exp(-delta/T) for temperature T
 *   - T decreases over time (cooling schedule)
 *   - Logs every experiment and hot-swaps the best config
 *
 * Run: node bench/experiment-harness.js [--experiments N] [--topK K]
 * Output: bench/experiment-results.json
 *
 * Evaluation:
 *   - Runs N scoring trials on the built-in test corpus
 *   - Quality score = Pearson correlation between expected relevance and computed score
 *   - Speed score = ops/sec for keyword.score over the corpus
 *   - Combined score = 0.5 * quality + 0.5 * normalised(speed)
 */

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const keyword = require('../src/keyword');
const semantic = require('../src/semantic');

// ── Ground-truth test corpus ─────────────────────────────────────────────────
// Each entry: { practSkills, reqSkills, niceSkills, expectedScore }
// expectedScore is a human-defined relevance: 1.0=perfect, 0.0=no match

const CORPUS = [
  {
    practSkills: ['React', 'TypeScript', 'Node.js', 'AWS', 'Docker'],
    reqSkills:   ['React', 'TypeScript', 'AWS'],
    niceSkills:  ['Docker', 'Kubernetes'],
    expectedScore: 0.95,  // perfect required match + one nice
  },
  {
    practSkills: ['Python', 'TensorFlow', 'scikit-learn', 'Pandas'],
    reqSkills:   ['Python', 'Machine Learning', 'TensorFlow'],
    niceSkills:  ['PyTorch', 'Keras'],
    expectedScore: 0.80,  // strong match but no PyTorch/Keras
  },
  {
    practSkills: ['Java', 'Spring Boot', 'Kafka', 'PostgreSQL'],
    reqSkills:   ['React', 'TypeScript', 'AWS'],
    niceSkills:  ['Docker'],
    expectedScore: 0.05,  // near-zero match (Java dev vs React role)
  },
  {
    practSkills: ['AWS', 'Terraform', 'Kubernetes', 'Docker'],
    reqSkills:   ['AWS', 'Kubernetes', 'Terraform'],
    niceSkills:  ['Ansible', 'Helm'],
    expectedScore: 0.90,  // all required matched
  },
  {
    practSkills: ['TypeScript', 'React', 'GraphQL', 'Node.js'],
    reqSkills:   ['React', 'TypeScript'],
    niceSkills:  ['GraphQL', 'Redis'],
    expectedScore: 0.92,  // perfect match + GraphQL bonus
  },
  {
    practSkills: ['C#', '.NET', 'Azure', 'SQL Server'],
    reqSkills:   ['Python', 'Django', 'PostgreSQL'],
    niceSkills:  ['Docker'],
    expectedScore: 0.10,  // minimal overlap
  },
  {
    practSkills: ['Kubernetes', 'Docker', 'CI/CD', 'Jenkins', 'AWS'],
    reqSkills:   ['Kubernetes', 'Docker', 'CI/CD'],
    niceSkills:  ['Terraform', 'AWS'],
    expectedScore: 0.92,  // excellent match
  },
  {
    practSkills: ['React', 'Angular', 'Vue', 'JavaScript'],
    reqSkills:   ['React', 'TypeScript'],
    niceSkills:  ['Redux'],
    expectedScore: 0.60,  // React present, no TypeScript
  },
];

// ── Algorithm names (match DEFAULT_WEIGHTS keys) ──────────────────────────────

const ALGO_KEYS = Object.keys(keyword.DEFAULT_WEIGHTS);

// ── Evaluation function ───────────────────────────────────────────────────────

/**
 * Score a weight configuration against the corpus.
 * Returns { quality: 0-1, speedOpsPerSec, combined: 0-1 }
 */
function evaluate(weights, threshold = 60) {
  const opts = { weights, matchThreshold: threshold };

  // Quality: Pearson correlation between expected and actual normalised scores
  const n = CORPUS.length;
  const actuals = CORPUS.map(c => keyword.score(c.practSkills, c.reqSkills, c.niceSkills, opts).score / 100);
  const expected = CORPUS.map(c => c.expectedScore);

  let sumA = 0, sumE = 0, sumAE = 0, sumA2 = 0, sumE2 = 0;
  for (let i = 0; i < n; i++) {
    sumA  += actuals[i];
    sumE  += expected[i];
    sumAE += actuals[i] * expected[i];
    sumA2 += actuals[i] * actuals[i];
    sumE2 += expected[i] * expected[i];
  }
  const meanA = sumA / n, meanE = sumE / n;
  let cov = 0, varA = 0, varE = 0;
  for (let i = 0; i < n; i++) {
    cov  += (actuals[i] - meanA) * (expected[i] - meanE);
    varA += (actuals[i] - meanA) ** 2;
    varE += (expected[i] - meanE) ** 2;
  }
  const denom = Math.sqrt(varA) * Math.sqrt(varE);
  const pearson = denom > 0 ? cov / denom : 0;
  const quality = (pearson + 1) / 2;  // map -1..1 → 0..1

  // Speed: ops/sec for the corpus
  const SPEED_ITERS = 500;
  const t0 = performance.now();
  for (let i = 0; i < SPEED_ITERS; i++) {
    for (const c of CORPUS) keyword.score(c.practSkills, c.reqSkills, c.niceSkills, opts);
  }
  const elapsed = performance.now() - t0;
  const speedOpsPerSec = Math.round((SPEED_ITERS * CORPUS.length) / (elapsed / 1000));

  // Normalise speed: baseline ~5700 ops/sec
  const normSpeed = Math.min(1, speedOpsPerSec / 10000);

  return {
    quality,
    speedOpsPerSec,
    combined: 0.5 * quality + 0.5 * normSpeed,
  };
}

// ── Random weight generation ──────────────────────────────────────────────────

function randomWeights(rng) {
  const w = {};
  for (const k of ALGO_KEYS) w[k] = Math.max(0.01, rng());
  // Normalise
  const total = Object.values(w).reduce((s, v) => s + v, 0);
  for (const k of ALGO_KEYS) w[k] = w[k] / total;
  return w;
}

function mutateWeights(weights, stepSize, rng) {
  const w = { ...weights };
  // Randomly perturb 2-3 weights
  const keysToMutate = ALGO_KEYS.filter(() => rng() < 0.5).slice(0, 3);
  if (keysToMutate.length === 0) keysToMutate.push(ALGO_KEYS[Math.floor(rng() * ALGO_KEYS.length)]);
  for (const k of keysToMutate) {
    w[k] = Math.max(0.001, w[k] + (rng() - 0.5) * stepSize);
  }
  // Normalise
  const total = Object.values(w).reduce((s, v) => s + Math.max(0, v), 0);
  for (const k of ALGO_KEYS) w[k] = Math.max(0, w[k]) / total;
  return w;
}

// ── Simple seeded PRNG (xoshiro128+) ─────────────────────────────────────────

function makePRNG(seed) {
  let a = seed | 0, b = seed ^ 0x9e3779b9, c = (seed << 5) | (seed >>> 27), d = seed * 0x6c62272e;
  return function() {
    const t = b << 9;
    let r = a * 5; r = (r << 7 | r >>> 25) * 9;
    c ^= a; d ^= b; b ^= c; a ^= d; c ^= t;
    d = d << 11 | d >>> 21;
    return (r >>> 0) / 0x100000000;
  };
}

// ── Main experiment loop ──────────────────────────────────────────────────────

const args          = process.argv.slice(2);
const _expIdx       = args.indexOf('--experiments');
const _topKIdx      = args.indexOf('--topK');
const N_EXPERIMENTS = _expIdx  >= 0 ? parseInt(args[_expIdx  + 1]) : 200;
const TOP_K         = _topKIdx >= 0 ? parseInt(args[_topKIdx + 1]) : 10;

console.log(`\n@futurenow/inference — Experiment Harness`);
console.log(`Experiments: ${N_EXPERIMENTS} | Corpus: ${CORPUS.length} entries\n`);

const experiments = [];
let bestConfig   = null;
let bestScore    = -Infinity;

// Simulated annealing schedule
const T_INIT     = 0.3;
const T_FINAL    = 0.01;
const COOLING    = Math.pow(T_FINAL / T_INIT, 1 / N_EXPERIMENTS);

// Restart every 40 experiments from the best so far
const RESTART_EVERY = 40;

let currentWeights   = keyword.DEFAULT_WEIGHTS;
let currentThreshold = 60;
let temperature      = T_INIT;
let rng              = makePRNG(42);

// Evaluate baseline
{
  const baseline = evaluate(keyword.DEFAULT_WEIGHTS, 60);
  bestConfig  = { weights: { ...keyword.DEFAULT_WEIGHTS }, threshold: 60 };
  bestScore   = baseline.combined;
  experiments.push({
    id: 0, type: 'baseline',
    weights: bestConfig.weights, threshold: 60,
    ...baseline,
    accepted: true, isBest: true,
  });
  console.log(`Baseline: quality=${baseline.quality.toFixed(3)} speed=${baseline.speedOpsPerSec} combined=${baseline.combined.toFixed(3)}`);
}

for (let i = 1; i <= N_EXPERIMENTS; i++) {
  // Restart from best every RESTART_EVERY iterations
  if (i % RESTART_EVERY === 0) {
    currentWeights   = { ...bestConfig.weights };
    currentThreshold = bestConfig.threshold;
    rng = makePRNG(i * 7919);  // new seed for diversity
  }

  // Generate next candidate
  const stepSize = 0.15 * Math.max(0.1, temperature / T_INIT);  // shrink step with temperature
  const candidateWeights   = mutateWeights(currentWeights, stepSize, rng);
  const candidateThreshold = Math.round(Math.max(40, Math.min(80, currentThreshold + (rng() - 0.5) * 10)));

  const result = evaluate(candidateWeights, candidateThreshold);

  // Metropolis acceptance criterion
  const delta    = result.combined - bestScore;
  const accepted = delta > 0 || Math.random() < Math.exp(delta / temperature);
  const isBest   = result.combined > bestScore;

  if (isBest) {
    bestConfig = { weights: candidateWeights, threshold: candidateThreshold };
    bestScore  = result.combined;
    process.stdout.write(`  🏆 New best at exp ${i}: combined=${result.combined.toFixed(3)} quality=${result.quality.toFixed(3)} speed=${result.speedOpsPerSec}\n`);
  }

  if (accepted) {
    currentWeights   = candidateWeights;
    currentThreshold = candidateThreshold;
  }

  experiments.push({
    id: i, type: 'annealing',
    weights: candidateWeights, threshold: candidateThreshold,
    ...result,
    temperature, accepted, isBest,
  });

  temperature *= COOLING;
}

// ── Report top-K configurations ──────────────────────────────────────────────

const topK = experiments
  .slice()
  .sort((a, b) => b.combined - a.combined)
  .slice(0, TOP_K);

console.log(`\nTop ${TOP_K} configurations:`);
console.log(`${'ID'.padEnd(5)} ${'Combined'.padEnd(10)} ${'Quality'.padEnd(10)} ${'Speed'.padEnd(10)} ${'Threshold'.padEnd(10)}`);
console.log('─'.repeat(55));
for (const e of topK) {
  console.log(
    String(e.id).padEnd(5) +
    e.combined.toFixed(4).padEnd(10) +
    e.quality.toFixed(4).padEnd(10) +
    String(e.speedOpsPerSec).padEnd(10) +
    String(e.threshold).padEnd(10)
  );
}

console.log(`\nBest configuration (experiment #${topK[0].id}):`);
console.log(JSON.stringify(bestConfig.weights, null, 2));
console.log(`Threshold: ${bestConfig.threshold}`);

// ── Write results ─────────────────────────────────────────────────────────────

const outputPath = path.join(__dirname, 'experiment-results.json');
fs.writeFileSync(outputPath, JSON.stringify({
  timestamp:   new Date().toISOString(),
  experiments: N_EXPERIMENTS,
  corpusSize:  CORPUS.length,
  baseline:    experiments[0],
  bestConfig,
  bestScore,
  topK,
  all:         experiments,
}, null, 2));

console.log(`\nFull results written to: ${outputPath}\n`);
