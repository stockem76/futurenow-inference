'use strict';
/**
 * sidecar.js — Sidecar process lifecycle for @futurenow/inference.
 *
 * Manages llama-cpp-python (LLM, port 52000) and OpenVINO embed server
 * (embed, port 52001) sidecar processes. Smart detection: if a port is
 * already listening (e.g. because SupplyMatch is running), the sidecar
 * is reused without spawning a new process.
 *
 * Exports:
 *   isRunning(port)      → Promise<boolean>
 *   start(type)          → Promise<void>  type: 'llm'|'embed'|'both'
 *   stop(type)           → Promise<void>
 *   _setConfig(cfg)      → called by index.js after init()
 */

const net    = require('net');
const { spawn } = require('child_process');
const path   = require('path');
const fs     = require('fs');

let _config = null;
function _setConfig(cfg) { _config = cfg; }

// ── Tracked child processes ───────────────────────────────────────────────────

const _procs = { llm: null, embed: null };

// ── TCP port probe ────────────────────────────────────────────────────────────

/**
 * Probe whether something is listening on port at 127.0.0.1.
 * Resolves true if the connection succeeds within 300ms, false otherwise.
 * @param {number} port
 * @returns {Promise<boolean>}
 */
function isRunning(port) {
  return new Promise(resolve => {
    const socket = net.createConnection({ port, host: '127.0.0.1' });
    const timeout = setTimeout(() => { socket.destroy(); resolve(false); }, 300);
    socket.on('connect', () => { clearTimeout(timeout); socket.destroy(); resolve(true); });
    socket.on('error',   () => { clearTimeout(timeout); resolve(false); });
  });
}

// ── Sidecar start/stop ────────────────────────────────────────────────────────

/**
 * Poll until a port is listening or timeout elapses.
 * @param {number} port
 * @param {number} maxMs
 * @returns {Promise<boolean>}
 */
function _waitForPort(port, maxMs) {
  return new Promise(resolve => {
    const deadline = Date.now() + maxMs;
    function check() {
      isRunning(port).then(up => {
        if (up) return resolve(true);
        if (Date.now() >= deadline) return resolve(false);
        setTimeout(check, 600);
      });
    }
    check();
  });
}

/**
 * Start the LLM sidecar (llama-cpp-python OpenAI-compatible server).
 * No-op if already running. Rejects if it doesn't become ready in 120s.
 */
async function _startLlm() {
  if (!_config) throw new Error('@futurenow/inference: call init() before sidecar.start()');
  const port = _config.llmPort || 52000;

  if (await isRunning(port)) return;  // already up (reuse SupplyMatch sidecar or prior start)

  const python  = _config.pythonBin || 'python';
  const models  = require('./models');
  const active  = models.getActive() || models.recommend({ task: 'generate' });
  const modelPath = active.modelPath || models.getPath(active.id);

  if (!modelPath || !fs.existsSync(modelPath)) {
    throw new Error(`@futurenow/inference: model file not found for "${active.id}" — check modelDir`);
  }

  const args = [
    '-m', 'llama_cpp.server',
    '--model', modelPath,
    '--host', '127.0.0.1',
    '--port', String(port),
    '--n_ctx', '4096',
  ];

  const proc = spawn(python, args, { stdio: 'ignore', detached: false });
  proc.on('error', err => { /* swallow — will surface as timeout */ });
  _procs.llm = proc;

  const ready = await _waitForPort(port, 120_000);
  if (!ready) {
    try { proc.kill('SIGTERM'); } catch (_) {}
    _procs.llm = null;
    throw new Error(`@futurenow/inference: LLM sidecar did not become ready within 120s on port ${port}`);
  }
}

/**
 * Start the embed sidecar (OpenVINO embed server).
 * Looks for embed_server.py in the inference module directory, then falls back
 * to a location passed via config.embedScript.
 */
async function _startEmbed() {
  if (!_config) throw new Error('@futurenow/inference: call init() before sidecar.start()');
  const port = _config.embedPort || 52001;

  if (await isRunning(port)) return;  // already up

  const python = _config.pythonBin || 'python';
  // Look for embed_server.py: config override → module-local copy → error
  const scriptCandidates = [
    _config.embedScript,
    path.join(__dirname, '..', 'python', 'embed_server.py'),
    path.join(__dirname, 'embed_server.py'),
  ].filter(Boolean);

  let script = null;
  for (const c of scriptCandidates) {
    if (fs.existsSync(c)) { script = c; break; }
  }
  if (!script) {
    throw new Error(
      '@futurenow/inference: embed_server.py not found. ' +
      'Place it in futurenow-inference/python/ or set opts.embedScript in init().'
    );
  }

  const proc = spawn(python, [script, '--port', String(port)], { stdio: 'ignore', detached: false });
  proc.on('error', () => {});
  _procs.embed = proc;

  const ready = await _waitForPort(port, 60_000);
  if (!ready) {
    try { proc.kill('SIGTERM'); } catch (_) {}
    _procs.embed = null;
    throw new Error(`@futurenow/inference: embed sidecar did not become ready within 60s on port ${port}`);
  }
}

/**
 * Start one or both sidecars. Reuses any already-running process.
 * @param {'llm'|'embed'|'both'} type
 */
async function start(type = 'both') {
  if (type === 'llm'  || type === 'both') await _startLlm();
  if (type === 'embed'|| type === 'both') await _startEmbed();
}

/**
 * Stop tracked sidecar process(es).
 * @param {'llm'|'embed'|'both'} type
 */
async function stop(type = 'both') {
  if ((type === 'llm' || type === 'both') && _procs.llm) {
    try { _procs.llm.kill('SIGTERM'); } catch (_) {}
    _procs.llm = null;
  }
  if ((type === 'embed' || type === 'both') && _procs.embed) {
    try { _procs.embed.kill('SIGTERM'); } catch (_) {}
    _procs.embed = null;
  }
}

// Clean up on process exit
process.on('exit', () => stop('both').catch(() => {}));

module.exports = { isRunning, start, stop, _setConfig };
