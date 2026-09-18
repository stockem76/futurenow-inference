'use strict';
/**
 * hardware.test.js — unit tests for @futurenow/inference src/hardware.js
 *
 * Tests the shape/contract of the detect() and getProvider() exports.
 * Actual NPU/GPU values are platform-specific; we verify types and structure
 * rather than specific hardware presence.
 */

const { ok, equal } = global._assert;

const hardware = require('../src/hardware');

module.exports = function () {

  // ── detect() structure ─────────────────────────────────────────────────────
  const info = hardware.detect();

  ok(typeof info === 'object' && info !== null,            'detect() returns an object');
  ok(info.cpu === true,                                    'info.cpu is always true');
  ok(typeof info.gpu          === 'boolean',               'info.gpu is a boolean');
  ok(typeof info.npu          === 'boolean',               'info.npu is a boolean');
  ok(typeof info.cuda         === 'boolean',               'info.cuda is a boolean');
  ok(typeof info.dml          === 'boolean',               'info.dml is a boolean');
  ok(typeof info.appleSilicon === 'boolean',               'info.appleSilicon is a boolean');
  ok(typeof info.intelNpu     === 'boolean',               'info.intelNpu is a boolean');
  ok(typeof info.platform     === 'string',                'info.platform is a string');
  ok(typeof info.arch         === 'string',                'info.arch is a string');
  ok(typeof info.cpuModel     === 'string',                'info.cpuModel is a string');
  ok(['webgpu', 'cpu'].includes(info.selectedProvider),   'selectedProvider is webgpu or cpu');

  // ── getProvider() ─────────────────────────────────────────────────────────
  const provider = hardware.getProvider();
  ok(['webgpu', 'cpu'].includes(provider),                'getProvider() returns webgpu or cpu');
  equal(provider, info.selectedProvider,                  'getProvider() matches detect().selectedProvider');

  // ── detect() is cached / idempotent ───────────────────────────────────────
  const info2 = hardware.detect();
  equal(info2.platform,      info.platform,               'second detect() call: platform stable');
  equal(info2.cpuModel,      info.cpuModel,               'second detect() call: cpuModel stable');
  equal(info2.selectedProvider, info.selectedProvider,    'second detect() call: selectedProvider stable');

  // ── _resetCache() clears the cached result ────────────────────────────────
  hardware._resetCache();
  const info3 = hardware.detect();
  ok(typeof info3 === 'object',                           '_resetCache() + detect() still returns object');
  ok(typeof info3.selectedProvider === 'string',          '_resetCache() + detect() has selectedProvider');

};
