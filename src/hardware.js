'use strict';
/**
 * hardware.js — CPU / GPU / NPU detection for @futurenow/inference.
 *
 * Adapted from SupplyMatch runtimeService.js. No SupplyMatch imports.
 * All probes are purely file-system / env checks — no child_process spawning.
 * Detection is cached after the first call (lazy singleton).
 *
 * Exports:
 *   detect()       → { cpu, gpu, npu, cuda, rocm, dml, opencl, openVino,
 *                       intelNpu, appleSilicon, platform, arch, cpuModel,
 *                       cpuCores, cpuThreads, ramGb, simd, selectedProvider,
 *                       recommendedThreads, onnxProviders }
 *   getProvider()  → 'webgpu' | 'cpu'
 *   bestOnnxProvider() → 'DmlExecutionProvider' | 'OpenVINOExecutionProvider' |
 *                         'CUDAExecutionProvider' | 'ROCMExecutionProvider' |
 *                         'CPUExecutionProvider'
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

function _fileExists(p) {
  try { return fs.existsSync(p); } catch (_) { return false; }
}

function _probeHardware() {
  const platform = process.platform;
  const arch     = process.arch;
  const cpuInfo  = os.cpus();
  const cpuModel = (cpuInfo[0] || {}).model || '';
  const cpuCores = os.cpus().filter((_, i) => i % 2 === 0).length || cpuInfo.length;  // physical cores estimate
  const cpuThreads = cpuInfo.length;                                                    // logical threads
  const ramGb    = Math.round(os.totalmem() / (1024 ** 3));

  // ── SIMD detection (V8 exposes WebAssembly SIMD as of Node 16) ────────────
  let simd = false;
  try {
    // Check for WASM SIMD support (Node 16+, all modern platforms)
    if (typeof WebAssembly !== 'undefined' &&
        arch !== 'ia32' &&
        process.versions.v8) {
      // x64 and arm64 always have SIMD on Node >= 16
      simd = ['x64', 'arm64', 'arm'].includes(arch);
    }
  } catch (_) { /* ignore */ }

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
      if (resolved && _fileExists(resolved)) { hasCuda = true; break; }
    } catch (_) { /* ignore */ }
  }

  // ── ROCm probe (AMD GPU on Linux) ─────────────────────────────────────────
  let hasRocm = false;
  if (platform === 'linux') {
    const rocmPaths = [
      '/opt/rocm/lib/librocm_smi64.so',
      '/opt/rocm/bin/rocminfo',
      '/usr/bin/rocm-smi',
    ];
    hasRocm = rocmPaths.some(_fileExists);
  }

  // ── DirectML probe (Intel / AMD on Windows) ───────────────────────────────
  let hasDml = false;
  if (platform === 'win32') {
    const dmlPaths = [
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'DirectML.dll'),
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'dml.dll'),
    ];
    hasDml = dmlPaths.some(_fileExists);
  }

  // ── OpenCL probe (cross-platform) ─────────────────────────────────────────
  let hasOpenCL = false;
  if (platform === 'win32') {
    const oclPaths = [
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenCL.dll'),
      path.join(process.env.SystemRoot || 'C:\\Windows', 'SysWOW64', 'OpenCL.dll'),
    ];
    hasOpenCL = oclPaths.some(_fileExists);
  } else if (platform === 'linux') {
    hasOpenCL = ['/usr/lib/libOpenCL.so', '/usr/lib/x86_64-linux-gnu/libOpenCL.so.1'].some(_fileExists);
  } else if (platform === 'darwin') {
    hasOpenCL = _fileExists('/System/Library/Frameworks/OpenCL.framework/OpenCL');
  }

  // ── OpenVINO probe (Intel NPU/GPU acceleration) ───────────────────────────
  let hasOpenVino = false;
  if (platform === 'win32') {
    const ovPaths = [
      path.join(process.env.PROGRAMFILES || 'C:\\Program Files', 'Intel', 'OpenVINO', 'bin'),
      path.join(process.env.PROGRAMFILES || 'C:\\Program Files', 'Intel', 'openvino_2024'),
      path.join(process.env.LOCALAPPDATA || '', 'openvino'),
    ];
    hasOpenVino = ovPaths.some(_fileExists);
    // Also check Python-installed OpenVINO (pip install openvino)
    if (!hasOpenVino) {
      const pyOvPaths = [
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'Lib', 'site-packages', 'openvino'),
        path.join(process.env.APPDATA || '', '..', 'Local', 'Programs', 'Python', 'Python311', 'Lib', 'site-packages', 'openvino'),
      ];
      hasOpenVino = pyOvPaths.some(_fileExists);
    }
  } else if (platform === 'linux') {
    hasOpenVino = [
      '/opt/intel/openvino',
      '/usr/local/lib/python3.11/dist-packages/openvino',
      '/usr/lib/python3/dist-packages/openvino',
    ].some(_fileExists);
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
      if (_fileExists(legacyDll)) hasIntelNpu = true;
    }

    // Core Ultra CPU model string implies NPU
    if (!hasIntelNpu && /Core\s+Ultra/i.test(cpuModel)) {
      hasIntelNpu = true;
    }
  }

  const gpu = hasCuda || hasRocm || hasDml || isAppleSilicon;
  const npu = isAppleSilicon || hasIntelNpu;

  // ── ONNX Runtime execution provider priority list ─────────────────────────
  // Ordered by expected performance on this machine (best first)
  const onnxProviders = [];
  if (hasCuda)        onnxProviders.push('CUDAExecutionProvider');
  if (hasRocm)        onnxProviders.push('ROCMExecutionProvider');
  if (hasIntelNpu && hasOpenVino) onnxProviders.push('OpenVINOExecutionProvider');
  if (hasDml)         onnxProviders.push('DmlExecutionProvider');
  if (isAppleSilicon) onnxProviders.push('CoreMLExecutionProvider');
  if (hasOpenCL)      onnxProviders.push('OpenCLExecutionProvider');
  onnxProviders.push('CPUExecutionProvider');

  // ── Recommended thread count for CPU inference ────────────────────────────
  // Use physical cores for compute threads (avoid SMT overhead for SIMD matmul)
  const recommendedThreads = Math.max(1, Math.min(cpuCores, 16));

  return {
    cpu: true,
    gpu,
    npu,
    cuda:        hasCuda,
    rocm:        hasRocm,
    dml:         hasDml,
    opencl:      hasOpenCL,
    openVino:    hasOpenVino,
    appleSilicon: isAppleSilicon,
    intelNpu:    hasIntelNpu,
    platform,
    arch,
    cpuModel:    cpuModel.slice(0, 60),
    cpuCores,
    cpuThreads,
    ramGb,
    simd,
    selectedProvider: (gpu || npu) ? PROVIDER_WEBGPU : PROVIDER_CPU,
    recommendedThreads,
    onnxProviders,
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Detect hardware capabilities. Result is cached after first call.
 * @returns {{ cpu, gpu, npu, cuda, rocm, dml, opencl, openVino,
 *             intelNpu, appleSilicon, platform, arch, cpuModel,
 *             cpuCores, cpuThreads, ramGb, simd,
 *             selectedProvider, recommendedThreads, onnxProviders }}
 */
function detect() {
  if (_cache) return _cache;
  try {
    _cache = _probeHardware();
  } catch (e) {
    _cache = {
      cpu: true, gpu: false, npu: false,
      cuda: false, rocm: false, dml: false, opencl: false, openVino: false,
      appleSilicon: false, intelNpu: false,
      platform: process.platform, arch: process.arch, cpuModel: '',
      cpuCores: Math.max(1, os.cpus().length),
      cpuThreads: os.cpus().length,
      ramGb: Math.round(os.totalmem() / (1024 ** 3)),
      simd: false,
      selectedProvider: PROVIDER_CPU,
      recommendedThreads: Math.max(1, Math.min(os.cpus().length, 8)),
      onnxProviders: ['CPUExecutionProvider'],
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

/**
 * Returns the best ONNX Runtime execution provider name for this machine.
 * Suitable for passing to onnxruntime-node session options.
 * @returns {string}
 */
function bestOnnxProvider() {
  const info = detect();
  return info.onnxProviders[0] || 'CPUExecutionProvider';
}

/** Reset the cached detection result (useful in tests). */
function _resetCache() { _cache = null; }

module.exports = { detect, getProvider, bestOnnxProvider, _resetCache };
