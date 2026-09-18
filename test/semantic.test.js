'use strict';
/**
 * semantic.test.js — unit tests for @futurenow/inference src/semantic.js
 *
 * Pure math — no I/O, no init() required.
 */

const { ok, equal, near } = global._assert;

const { cosine, normalize, composite } = require('../src/semantic');

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeVec(values) { return new Float32Array(values); }

/** L2-normalise a plain array, return Float32Array */
function unit(values) {
  const mag = Math.sqrt(values.reduce((s, v) => s + v * v, 0));
  return makeVec(values.map(v => v / mag));
}

module.exports = function () {

  // ── cosine ────────────────────────────────────────────────────────────────

  const a  = unit([1, 0, 0]);
  const b  = unit([1, 0, 0]);
  const c  = unit([0, 1, 0]);
  const d  = unit([-1, 0, 0]);
  const ab = unit([0.707, 0.707, 0]);

  near(cosine(a, b), 1.0,  0.001,  'identical unit vectors → cosine ≈ 1.0');
  near(cosine(a, c), 0.0,  0.001,  'orthogonal unit vectors → cosine ≈ 0.0');
  near(cosine(a, d), -1.0, 0.001,  'opposite unit vectors → cosine ≈ -1.0');
  near(cosine(a, ab), 0.707, 0.01, '45° angle → cosine ≈ 0.707');

  // Null / mismatched inputs return 0
  equal(cosine(null,  a),    0,    'null first arg → 0');
  equal(cosine(a,     null), 0,    'null second arg → 0');
  equal(cosine(a, makeVec([1, 0])), 0, 'mismatched length → 0');

  // ── normalize ────────────────────────────────────────────────────────────

  // At centre (0.70) output should be 50
  near(normalize(0.70), 50, 2,    'cosine at centre (0.70) → ~50');

  // High cosine should give high score
  ok(normalize(0.90) > 70,        'cosine 0.90 → score > 70');

  // Low cosine should give low score
  ok(normalize(0.40) < 20,        'cosine 0.40 → score < 20');

  // Boundary: cosine = 1.0 should give score ≤ 100
  ok(normalize(1.0) <= 100,       'cosine 1.0 → score ≤ 100');
  ok(normalize(-1.0) >= 0,        'cosine -1.0 → score ≥ 0');

  // asPercent = false returns 0-1
  const raw = normalize(0.70, { asPercent: false });
  ok(raw > 0 && raw < 1,          'asPercent:false → value in (0,1)');

  // Custom centre shifts the midpoint
  const highCentre = normalize(0.90, { centre: 0.90 });
  near(highCentre, 50, 3,         'custom centre 0.90 → output ≈ 50 at cosine 0.90');

  // ── composite ────────────────────────────────────────────────────────────

  // 50/50 blend
  near(composite(60, 80, 0.5, 0.5), 70, 1,  'composite 50/50 of 60+80 → 70');

  // Full weight to semantic
  equal(composite(60, 80, 1.0, 0.0),  60,   'α=1, β=0 → semantic only');

  // Full weight to keyword
  equal(composite(60, 80, 0.0, 1.0),  80,   'α=0, β=1 → keyword only');

  // Default weights α=0.40 β=0.60
  const def = composite(100, 0);
  equal(def, 40,                              'default α=0.40: composite(100,0) → 40');

  // Bounded output
  ok(composite(200, 200) <= 100,             'composite clamped to 100');
  ok(composite(-50, -50) >= 0,               'composite clamped to 0');

};
