'use strict';
/**
 * llmClient.js — LLM text generation via llama-cpp-python OpenAI-compatible sidecar.
 *
 * POSTs to http://127.0.0.1:{llmPort}/v1/chat/completions.
 * Supports both streamed (SSE) and non-streamed responses.
 * Returns null on any failure — never throws.
 *
 * Exports:
 *   complete(messages, opts?)      → Promise<string|null>
 *   stream(messages, opts, onChunk) → Promise<void>
 *   isAvailable()                  → Promise<boolean>
 *   _setConfig(cfg)                → called by index.js after init()
 */

const http = require('http');

let _config = null;
function _setConfig(cfg) { _config = cfg; }

const DEFAULT_TIMEOUT_MS  = 120_000;   // 2 min — LLM can be slow on CPU
const PROBE_TIMEOUT_MS    = 3_000;

// ── Helpers ───────────────────────────────────────────────────────────────────

function _port() { return (_config && _config.llmPort) || 52000; }

/**
 * Fire a raw POST to the sidecar and collect the full response body.
 * @param {string}  path
 * @param {object}  payload
 * @param {number}  [timeoutMs]
 * @returns {Promise<{status:number, body:string}|null>}
 */
function _post(path, payload, timeoutMs = DEFAULT_TIMEOUT_MS) {
  return new Promise(resolve => {
    const body = Buffer.from(JSON.stringify(payload));
    const req  = http.request(
      {
        hostname: '127.0.0.1',
        port:     _port(),
        path,
        method:   'POST',
        headers: {
          'Content-Type':   'application/json',
          'Content-Length': body.length,
        },
        timeout: timeoutMs,
      },
      res => {
        let raw = '';
        res.on('data',  d  => { raw += d; });
        res.on('end',   ()  => resolve({ status: res.statusCode, body: raw }));
        res.on('error', ()  => resolve(null));
      }
    );
    req.on('error',   () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
    req.write(body);
    req.end();
  });
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Probe whether the LLM sidecar is responding.
 * Uses GET /v1/models — lightweight, no generation required.
 * @returns {Promise<boolean>}
 */
function isAvailable() {
  return new Promise(resolve => {
    const req = http.get(
      {
        hostname: '127.0.0.1',
        port:     _port(),
        path:     '/v1/models',
        timeout:  PROBE_TIMEOUT_MS,
      },
      res => {
        res.resume();                       // drain so socket closes
        resolve(res.statusCode === 200);
      }
    );
    req.on('error',   () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

/**
 * Non-streaming chat completion.
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {object}  [opts]
 * @param {number}  [opts.maxTokens=1024]
 * @param {number}  [opts.temperature=0.3]
 * @param {number}  [opts.topP=0.9]
 * @param {string}  [opts.model='local-model']
 * @param {number}  [opts.timeoutMs]
 * @returns {Promise<string|null>} assistant message text or null on failure
 */
async function complete(messages, opts = {}) {
  const payload = {
    model:       opts.model        || 'local-model',
    messages,
    max_tokens:  opts.maxTokens    || 1024,
    temperature: opts.temperature  ?? 0.3,
    top_p:       opts.topP         ?? 0.9,
    stream:      false,
  };

  const result = await _post('/v1/chat/completions', payload, opts.timeoutMs);
  if (!result || result.status !== 200) return null;

  try {
    const j = JSON.parse(result.body);
    return j.choices?.[0]?.message?.content ?? null;
  } catch (_) {
    return null;
  }
}

/**
 * Streaming chat completion — calls `onChunk(deltaText)` for each SSE token
 * as it arrives.  Resolves when the stream ends (or on error).
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {object}  [opts]
 * @param {number}  [opts.maxTokens=1024]
 * @param {number}  [opts.temperature=0.3]
 * @param {number}  [opts.topP=0.9]
 * @param {string}  [opts.model='local-model']
 * @param {number}  [opts.timeoutMs]
 * @param {function(string):void} onChunk — called with each text delta
 * @returns {Promise<void>}
 */
function stream(messages, opts = {}, onChunk = () => {}) {
  const payload = {
    model:       opts.model        || 'local-model',
    messages,
    max_tokens:  opts.maxTokens    || 1024,
    temperature: opts.temperature  ?? 0.3,
    top_p:       opts.topP         ?? 0.9,
    stream:      true,
  };

  return new Promise(resolve => {
    const body = Buffer.from(JSON.stringify(payload));
    const req  = http.request(
      {
        hostname: '127.0.0.1',
        port:     _port(),
        path:     '/v1/chat/completions',
        method:   'POST',
        headers: {
          'Content-Type':   'application/json',
          'Content-Length': body.length,
        },
        timeout: opts.timeoutMs || DEFAULT_TIMEOUT_MS,
      },
      res => {
        let buffer = '';

        res.on('data', chunk => {
          buffer += chunk.toString('utf8');
          // SSE lines arrive as "data: {...}\n\n"
          const lines = buffer.split('\n');
          buffer = lines.pop();              // keep incomplete trailing line

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data:')) continue;
            const jsonStr = trimmed.slice(5).trim();
            if (jsonStr === '[DONE]') continue;
            try {
              const j = JSON.parse(jsonStr);
              const delta = j.choices?.[0]?.delta?.content;
              if (delta) onChunk(delta);
            } catch (_) { /* skip malformed lines */ }
          }
        });

        res.on('end',   () => resolve());
        res.on('error', () => resolve());
      }
    );
    req.on('error',   () => resolve());
    req.on('timeout', () => { req.destroy(); resolve(); });
    req.write(body);
    req.end();
  });
}

module.exports = { complete, stream, isAvailable, _setConfig };
