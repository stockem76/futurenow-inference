'use strict';
/**
 * hardware-v2.test.js — Tests for new hardware.js v2 fields.
 *
 * Tests: cpuCores, cpuThreads, ramGb, simd, recommendedThreads,
 *        onnxProviders, bestOnnxProvider().
 */

const { ok, equal } = global._assert;

const hardware = require('../src/hardware');

module.exports = function () {

  hardware._resetCache();
  const info = hardware.detect();

  // ── New v2 fields ─────────────────────────────────────────────────────────

  ok(typeof info.cpuCores         === 'number',            'info.cpuCores is number');
  ok(typeof info.cpuThreads       === 'number',            'info.cpuThreads is number');
  ok(typeof info.ramGb            === 'number',            'info.ramGb is number');
  ok(typeof info.simd             === 'boolean',           'info.simd is boolean');
  ok(typeof info.recommendedThreads === 'number',          'info.recommendedThreads is number');
  ok(Array.isArray(info.onnxProviders),                    'info.onnxProviders is Array');
  ok(typeof info.rocm             === 'boolean',           'info.rocm is boolean');
  ok(typeof info.opencl           === 'boolean',           'info.opencl is boolean');
  ok(typeof info.openVino         === 'boolean',           'info.openVino is boolean');

  // Values make sense
  ok(info.cpuCores >= 1,                                   'cpuCores ≥ 1');
  ok(info.cpuThreads >= info.cpuCores,                     'cpuThreads ≥ cpuCores');
  ok(info.ramGb >= 1,                                      'ramGb ≥ 1 GB');
  ok(info.recommendedThreads >= 1,                         'recommendedThreads ≥ 1');
  ok(info.recommendedThreads <= 16,                        'recommendedThreads ≤ 16 (cap)');
  ok(info.onnxProviders.length >= 1,                       'onnxProviders has ≥ 1 entry');

  // CPUExecutionProvider must always be in onnxProviders (fallback)
  ok(info.onnxProviders.includes('CPUExecutionProvider'),  'onnxProviders contains CPUExecutionProvider');

  // All provider names are strings
  for (const p of info.onnxProviders) {
    ok(typeof p === 'string',                              `onnxProvider ${p} is a string`);
  }

  // ── bestOnnxProvider() ────────────────────────────────────────────────────

  const best = hardware.bestOnnxProvider();
  ok(typeof best === 'string',                             'bestOnnxProvider() returns string');
  ok(best.length > 0,                                      'bestOnnxProvider() non-empty');
  ok(info.onnxProviders.includes(best),                    'bestOnnxProvider() is in onnxProviders list');

  // Calling again returns same result (cached)
  const best2 = hardware.bestOnnxProvider();
  equal(best, best2,                                       'bestOnnxProvider() is stable');

  // ── _resetCache() clears new fields too ──────────────────────────────────

  hardware._resetCache();
  const info2 = hardware.detect();
  ok(typeof info2.cpuCores === 'number',                   '_resetCache(): cpuCores still present');
  ok(typeof info2.onnxProviders === 'object',              '_resetCache(): onnxProviders still present');
  ok(info2.onnxProviders.includes('CPUExecutionProvider'), '_resetCache(): CPU provider still present');

};
