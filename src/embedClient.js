'use strict';
/**
 * embedClient.js — Text embedding via OpenVINO sidecar.
 *
 * POSTs to http://127.0.0.1:{embedPort}/embed.
 * Returns Float32Array vectors (384-dim for gte-small).
 * Returns null on any failure — never throws.
 *
 * Exports:
 *   single(text)    → Promise<Float32Array|null>
 *   batch(texts)    → Promise<Float32Array[]|null>
 *   _setConfig(cfg) → called by index.js after init()
 */

const http = require('http');

let _config = null;
function _setConfig(cfg) { _config = cfg; }

const TIMEOUT_MS = 30_000;

/**
 * POST texts array to the embed sidecar.
 * @param {string[]} texts
 * @returns {Promise<Float32Array[]|null>}
 */
async function batch(texts) {
  if (!texts || !texts.length) return [];
  const port = (_config && _config.embedPort) || 52001;
  const body = Buffer.from(JSON.stringify({ texts }));

  return new Promise(resolve => {
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port,
        path:     '/embed',
        method:   'POST',
        headers: {
          'Content-Type':   'application/json',
          'Content-Length': body.length,
        },
        timeout: TIMEOUT_MS,
      },
      res => {
        let raw = '';
        res.on('data', d => { raw += d; });
        res.on('end', () => {
          if (res.statusCode !== 200) { resolve(null); return; }
          try {
            const j = JSON.parse(raw);
            resolve(j.embeddings.map(e => new Float32Array(e)));
          } catch (_) { resolve(null); }
        });
      }
    );
    req.on('error',   () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
    req.write(body);
    req.end();
  });
}

/**
 * Embed a single text string.
 * @param {string} text
 * @returns {Promise<Float32Array|null>}
 */
async function single(text) {
  const results = await batch([text]);
  if (!results) return null;
  return results[0] || null;
}

module.exports = { single, batch, _setConfig };
