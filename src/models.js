'use strict';
/**
 * models.js — Model catalogue, recommendation, and selection.
 *
 * Hardcodes the five-model catalogue sourced from SupplyMatch's
 * local-llm-prompts.json recommendedModels. No file I/O at require time.
 *
 * Exports:
 *   catalogue()            → annotated model list (with available: boolean)
 *   recommend({ task, ramGb? }) → best model for the task + hardware
 *   select(id)             → set the active model for subsequent calls
 *   getActive()            → currently selected model spec | null
 *   _setConfig(cfg)        → called by index.js after init()
 */

const fs   = require('fs');
const path = require('path');
const os   = require('os');

// ── Model catalogue (source of truth) ─────────────────────────────────────────

const CATALOGUE = [
  {
    id:           'llama-3.2-3b-instruct-q4',
    name:         'Llama 3.2 3B Instruct Q4_K_M',
    description:  'Fast & light. Best for 16 GB machines, or any machine where responsiveness matters more than reasoning depth.',
    ramTier:      16,
    sizeGb:       2.0,
    ramRequiredGb: 8,
    tags:         ['fast', 'light', 'llama', '16gb-tier'],
    filename:     'llama-3.2-3b-instruct-q4.gguf',
    url:          'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
  },
  {
    id:           'llama-3.1-8b-instruct-q4',
    name:         'Meta Llama 3.1 8B Instruct Q4_K_M',
    description:  'Balanced default. Strong general reasoning — the recommended choice for 32 GB machines.',
    ramTier:      32,
    sizeGb:       4.9,
    ramRequiredGb: 16,
    tags:         ['default', 'balanced', 'llama', '32gb-tier'],
    filename:     'llama-3.1-8b-instruct-q4.gguf',
    url:          'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
  },
  {
    id:           'qwen-2.5-7b-instruct-q4',
    name:         'Qwen 2.5 7B Instruct Q4_K_M',
    description:  'Structured analyst. Best of the 7-8B tier for schema-adherent reasoning.',
    ramTier:      32,
    sizeGb:       4.7,
    ramRequiredGb: 16,
    tags:         ['analyst', 'structured', 'qwen', '32gb-tier'],
    filename:     'qwen-2.5-7b-instruct-q4.gguf',
    url:          'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
  },
  {
    id:           'mistral-7b-instruct-q4',
    name:         'Mistral 7B Instruct v0.3 Q4_K_M',
    description:  'Fast & direct. Lightest of the 7-8B tier for quick terse responses.',
    ramTier:      32,
    sizeGb:       4.4,
    ramRequiredGb: 16,
    tags:         ['fast', 'direct', 'mistral', '32gb-tier'],
    filename:     'mistral-7b-instruct-q4.gguf',
    url:          'https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
  },
  {
    id:           'qwen-2.5-14b-instruct-q4',
    name:         'Qwen 2.5 14B Instruct Q4_K_M',
    description:  'Deep reasoning. For 64 GB machines where response quality matters more than speed.',
    ramTier:      64,
    sizeGb:       9.0,
    ramRequiredGb: 32,
    tags:         ['deep', 'reasoning', 'qwen', '64gb-tier'],
    filename:     'qwen-2.5-14b-instruct-q4.gguf',
    url:          'https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf',
  },
];

// ── Module state ──────────────────────────────────────────────────────────────

let _config     = null;
let _activeId   = null;

function _setConfig(cfg) { _config = cfg; }

function _modelDir() {
  return _config && _config.modelDir;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Return the full catalogue, annotating each entry with available: boolean.
 * available is true when the .gguf file exists in modelDir.
 * @returns {Array}
 */
function catalogue() {
  const dir = _modelDir();
  return CATALOGUE.map(m => {
    let available = false;
    if (dir) {
      try { available = fs.existsSync(path.join(dir, m.filename)); } catch (_) { /* ignore */ }
    }
    return Object.assign({}, m, { available });
  });
}

/**
 * Recommend a model for the given task and available RAM.
 * @param {{ task: 'embed'|'generate', ramGb?: number }} opts
 * @returns {{ id, name, device: 'NPU'|'GPU'|'CPU', filename, available }}
 */
function recommend({ task = 'generate', ramGb } = {}) {
  const hw       = require('./hardware').detect();
  const device   = hw.npu ? 'NPU' : hw.gpu ? 'GPU' : 'CPU';
  const ram      = ramGb || Math.round(os.totalmem() / (1024 ** 3));
  const cat      = catalogue();

  if (task === 'embed') {
    // Embedding always uses the sidecar — just return a device descriptor
    return { id: 'embed-sidecar', name: 'OpenVINO Embed Sidecar', device, available: true };
  }

  // Pick the highest-RAM-tier model that fits available RAM and is present on disk
  const available = cat.filter(m => m.available && m.ramRequiredGb <= ram);
  if (available.length) {
    available.sort((a, b) => b.ramRequiredGb - a.ramRequiredGb);
    const pick = available[0];
    return { id: pick.id, name: pick.name, device, filename: pick.filename, available: true };
  }

  // Nothing available on disk — recommend smallest model that fits RAM
  const fits = [...CATALOGUE].filter(m => m.ramRequiredGb <= ram);
  fits.sort((a, b) => b.ramRequiredGb - a.ramRequiredGb);
  const fallback = fits[0] || CATALOGUE[0];
  return { id: fallback.id, name: fallback.name, device, filename: fallback.filename, available: false };
}

/**
 * Set the active model by id. Subsequent inference.llm.generate() calls
 * will use this model.
 * @param {string} id
 */
function select(id) {
  const found = CATALOGUE.find(m => m.id === id);
  if (!found) throw new Error(`@futurenow/inference: unknown model id "${id}"`);
  _activeId = id;
}

/**
 * Return the currently selected model spec, or null if none selected.
 */
function getActive() {
  if (!_activeId) return null;
  const dir = _modelDir();
  const m   = CATALOGUE.find(m => m.id === _activeId);
  if (!m) return null;
  let available = false;
  if (dir) {
    try { available = fs.existsSync(path.join(dir, m.filename)); } catch (_) { /* ignore */ }
  }
  return Object.assign({}, m, { available, modelPath: dir ? path.join(dir, m.filename) : null });
}

/**
 * Return the resolved path for a model by id. Null if modelDir not set.
 * @param {string} id
 */
function getPath(id) {
  const dir = _modelDir();
  if (!dir) return null;
  const m = CATALOGUE.find(m => m.id === id);
  if (!m) return null;
  return path.join(dir, m.filename);
}

module.exports = { catalogue, recommend, select, getActive, getPath, _setConfig, _CATALOGUE: CATALOGUE };
