'use strict';
/**
 * @futurenow/inference — FutureNow local AI inference module.
 *
 * Single entry point. Call inference.init(opts) once before using any
 * other namespace. All namespaces are lazy — they will throw a clear
 * error if init() has not been called.
 *
 * Namespaces:
 *   inference.hardware  — CPU/GPU/NPU detection
 *   inference.models    — model catalogue, recommend, select
 *   inference.sidecar   — start/stop llama-cpp-python + OpenVINO sidecar processes
 *   inference.embed     — single and batch text embedding via sidecar
 *   inference.keyword   — keyword scoring (Jaro-Winkler depth tiers)
 *   inference.semantic  — cosine, sigmoid normalise, composite blend
 *   inference.llm       — LLM text generation via local sidecar
 *   inference.cache     — 3-tier vector cache (hot Map + TTL + optional SQLite)
 */

const hardware = require('./src/hardware');
const models   = require('./src/models');
const sidecar  = require('./src/sidecar');
const embed    = require('./src/embedClient');
const keyword  = require('./src/keyword');
const semantic = require('./src/semantic');
const llm      = require('./src/llmClient');
const cache    = require('./src/cache');

// ── Module-level config ───────────────────────────────────────────────────────

let _config = null;

/**
 * Initialise the inference module. Must be called once before any other use.
 *
 * @param {object} opts
 * @param {string}  opts.modelDir       — directory containing .gguf model files (required)
 * @param {number}  [opts.embedPort=52001] — OpenVINO embed sidecar port
 * @param {number}  [opts.llmPort=52000]   — llama-cpp-python LLM sidecar port
 * @param {string}  [opts.coldStorePath]   — optional path for SQLite cold vector cache
 * @param {string}  [opts.pythonBin='python'] — Python binary for spawning sidecars
 * @param {string}  [opts.embedScript]    — path to embed_server.py (optional override)
 * @param {string}  [opts.llmScript]      — path to llm_server.py (optional override)
 */
function init(opts) {
  if (!opts || !opts.modelDir) throw new Error('@futurenow/inference: init() requires opts.modelDir');
  _config = {
    modelDir:      opts.modelDir,
    embedPort:     Number(opts.embedPort)  || 52001,
    llmPort:       Number(opts.llmPort)    || 52000,
    coldStorePath: opts.coldStorePath      || null,
    pythonBin:     opts.pythonBin          || 'python',
    embedScript:   opts.embedScript        || null,
    llmScript:     opts.llmScript          || null,
  };

  // Pass config into sub-modules that need it
  models._setConfig(_config);
  sidecar._setConfig(_config);
  embed._setConfig(_config);
  llm._setConfig(_config);
  cache._setConfig(_config);
}

/** Returns the active config or throws if init() has not been called. */
function _getConfig() {
  if (!_config) throw new Error('@futurenow/inference: call init(opts) before using this module');
  return _config;
}

// ── Public API ────────────────────────────────────────────────────────────────

module.exports = {
  init,
  _getConfig,  // exposed so sub-modules can call back for late-binding

  /**
   * CPU / GPU / NPU / OpenCL / OpenVINO / ROCm detection.
   * No init() required. Extended in v2: cpuCores, cpuThreads, ramGb, simd,
   * recommendedThreads, onnxProviders, bestOnnxProvider().
   */
  hardware,

  /** Model catalogue + recommend + select */
  models,

  /** Sidecar process lifecycle — requires init() */
  sidecar,

  /** Text embedding via sidecar — requires init() */
  embed,

  /**
   * Keyword scoring (pure, no I/O). Extended in v2:
   *   bm25Score(), tfidfPositional(), buildInvertedIndex(), proximityScore().
   */
  keyword,

  /**
   * Semantic scoring utilities (pure math). Extended in v2:
   *   cosineAll(), rankAll(), cosineBatch(), normalizeAll().
   */
  semantic,

  /**
   * LLM text generation via local sidecar — requires init(). Extended in v2:
   *   completeWithFallback(), parseStructured(), buildMatchPrompt(), buildAssessPrompt().
   */
  llm,

  /**
   * 3-tier vector cache — requires init(). Extended in v2:
   *   getMany(), setMany(), contentHash(), getOrEmbed().
   */
  cache,
};
