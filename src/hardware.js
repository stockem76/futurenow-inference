'use strict';
/**
 * hardware.js — CPU / GPU / NPU detection for @futurenow/inference.
 *
 * Adapted from SupplyMatch runtimeService.js. No SupplyMatch imports.
 * All probes are purely file-system / env checks — no child_process spawning.
 * Detection is cached after the first call (lazy singleton).
 *
 * Exports:
 *   detect()       → { cpu, gpu, npu, cuda, dml, intelNpu, appleSilicon,
 *                       platform, arch, cpuModel, selectedProvider }
 *   getProvider()  → 'webgpu' | 'cpu'
 */

const os   = require('os');
const path = require('path');
const fs   = require('fs');

const PROVIDER_WEBGPU = 'webgpu';
const PROVIDER_CPU    = 'cpu';

let _cache = null;

// ── Hardware probes ───────────────────────────────────────────────────────────

function _findInPath(name) {
  const dirs = (process.env.PATH || '').split(path.delimiter);
  for (const dir of dirs) {
    if (!dir) continue;
    const full = path.join(dir, name);
    try { if (fs.existsSync(full)) return full; } catch (_) { /* ignore */ }
    if (process.platform === 'win32' && !name.toLowerCase().endsWith('.exe')) {
      const fullExe = full + '.exe';
      try { if (fs.existsSync(fullExe)) return fullExe; } catch (_) { /* ignore */ }
    }
  }
  return null;
}

function _probeHardware() {
  const platform = process.platform;
  const arch     = process.arch;
  const cpuModel = (os.cpus()[0] || {}).model || '';

  // ── CUDA probe (NVIDIA GPU) ───────────────────────────────────────────────
  let hasCuda = false;
  const nvidiaCandidates = [
    'nvidia-smi',
    'C:\\Windows\\System32\\nvidia-smi.exe',
    'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
  ];
  for (const c of nvidiaCandidates) {
    try {
      const resolved = path.isAbsolute(c) ? c : _findInPath(c);
      if (resolved && fs.existsSync(resolved)) { hasCuda = true; break; }
    } catch (_) { /* ignore */ }
  }

  // ── DirectML probe (Intel / AMD on Windows) ───────────────────────────────
  let hasDml = false;
  if (platform === 'win32') {
    const dmlPaths = [
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'DirectML.dll'),
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'dml.dll'),
    ];
    for (const p of dmlPaths) {
      try { if (fs.existsSync(p)) { hasDml = true; break; } } catch (_) { /* ignore */ }
    }
  }

  // ── Apple Silicon / NPU probe ─────────────────────────────────────────────
  const isAppleSilicon = platform === 'darwin' && arch === 'arm64';

  // ── Intel NPU probe (AI PC — Meteor Lake / Lunar Lake / Arrow Lake) ───────
  let hasIntelNpu = false;
  if (platform === 'win32') {
    const driverStore = path.join(
      process.env.SystemRoot || 'C:\\Windows',
      'System32', 'DriverStore', 'FileRepository'
    );
    try {
      if (fs.existsSync(driverStore)) {
        const entries = fs.readdirSync(driverStore);
        hasIntelNpu = entries.some(e => /^npu\.inf_/i.test(e));
      }
    } catch (_) { /* ignore */ }

    if (!hasIntelNpu) {
      const legacyDll = path.join(
        process.env.SystemRoot || 'C:\\Windows', 'System32', 'intel_npu_driver.dll'
      );
      try { if (fs.existsSync(legacyDll)) hasIntelNpu = true; } catch (_) { /* ignore */ }
    }

    // Core Ultra CPU model string implies NPU
    if (!hasIntelNpu && /Core\s+Ultra/i.test(cpuModel)) {
      hasIntelNpu = true;
    }
  }

  const gpu = hasCuda || hasDml || isAppleSilicon;
  const npu = isAppleSilicon || hasIntelNpu;

  return {
    cpu: true,
    gpu,
    npu,
    cuda: hasCuda,
    dml:  hasDml,
    appleSilicon: isAppleSilicon,
    intelNpu: hasIntelNpu,
    platform,
    arch,
    cpuModel: cpuModel.slice(0, 60),
    selectedProvider: (gpu || npu) ? PROVIDER_WEBGPU : PROVIDER_CPU,
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Detect hardware capabilities. Result is cached after first call.
 * @returns {{ cpu, gpu, npu, cuda, dml, intelNpu, appleSilicon,
 *             platform, arch, cpuModel, selectedProvider }}
 */
function detect() {
  if (_cache) return _cache;
  try {
    _cache = _probeHardware();
  } catch (e) {
    _cache = {
      cpu: true, gpu: false, npu: false,
      cuda: false, dml: false, appleSilicon: false, intelNpu: false,
      platform: process.platform, arch: process.arch, cpuModel: '',
      selectedProvider: PROVIDER_CPU,
      detectionError: e.message,
    };
  }
  return _cache;
}

/**
 * Returns the recommended ONNX execution provider string.
 * @returns {'webgpu' | 'cpu'}
 */
function getProvider() {
  return detect().selectedProvider;
}

/** Reset the cached detection result (useful in tests). */
function _resetCache() { _cache = null; }

module.exports = { detect, getProvider, _resetCache };
