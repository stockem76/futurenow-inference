'use strict';
/**
 * llmClient.js — LLM text generation via llama-cpp-python OpenAI-compatible sidecar.
 *
 * POSTs to http://127.0.0.1:{llmPort}/v1/chat/completions.
 * Supports both streamed (SSE) and non-streamed responses.
 * Returns null on any failure — never throws.
 *
 * v2 improvements:
 *   - complete() retries once on transient errors (network hiccup resilience)
 *   - completeWithFallback() tries decreasing max_tokens if model context is exceeded
 *   - parseStructured() extracts structured sections from free-text LLM output
 *   - buildMatchPrompt() produces a high-quality, specific AI match prompt
 *   - buildAssessPrompt() produces a contextually rich assessment prompt
 *   - selectBestModel() uses hardware info to choose optimal model
 *
 * Exports:
 *   complete(messages, opts?)                → Promise<string|null>
 *   completeWithFallback(messages, opts?)    → Promise<string|null>
 *   stream(messages, opts, onChunk)         → Promise<void>
 *   isAvailable()                            → Promise<boolean>
 *   parseStructured(text, sections)          → object
 *   buildMatchPrompt(context)               → Array<{role,content}>
 *   buildAssessPrompt(context)              → Array<{role,content}>
 *   _setConfig(cfg)                          → called by index.js after init()
 */

const http = require('http');

let _config = null;
function _setConfig(cfg) { _config = cfg; }

const DEFAULT_TIMEOUT_MS  = 120_000;   // 2 min — LLM can be slow on CPU
const PROBE_TIMEOUT_MS    = 3_000;
const RETRY_DELAY_MS      = 500;

// ── Helpers ───────────────────────────────────────────────────────────────────

function _port() { return (_config && _config.llmPort) || 52000; }

/**
 * Fire a raw POST to the sidecar and collect the full response body.
 * @param {string}  reqPath
 * @param {object}  payload
 * @param {number}  [timeoutMs]
 * @returns {Promise<{status:number, body:string}|null>}
 */
function _post(reqPath, payload, timeoutMs = DEFAULT_TIMEOUT_MS) {
  return new Promise(resolve => {
    const body = Buffer.from(JSON.stringify(payload));
    const req  = http.request(
      {
        hostname: '127.0.0.1',
        port:     _port(),
        path:     reqPath,
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

/** Sleep for ms milliseconds */
function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Probe whether the LLM sidecar is responding.
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
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on('error',   () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

/**
 * Non-streaming chat completion with one automatic retry on transient failure.
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {object}  [opts]
 * @param {number}  [opts.maxTokens=1024]
 * @param {number}  [opts.temperature=0.3]
 * @param {number}  [opts.topP=0.9]
 * @param {string}  [opts.model='local-model']
 * @param {number}  [opts.timeoutMs]
 * @param {boolean} [opts.retry=true]
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

  let result = await _post('/v1/chat/completions', payload, opts.timeoutMs);
  if (!result || result.status !== 200) {
    // One retry after a short delay
    if (opts.retry !== false) {
      await _sleep(RETRY_DELAY_MS);
      result = await _post('/v1/chat/completions', payload, opts.timeoutMs);
    }
  }
  if (!result || result.status !== 200) return null;

  try {
    const j = JSON.parse(result.body);
    return j.choices?.[0]?.message?.content ?? null;
  } catch (_) {
    return null;
  }
}

/**
 * Complete with automatic fallback to smaller token budgets if the first call
 * fails or produces a context-length error. Tries maxTokens then halved values.
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {object} [opts]  — same as complete()
 * @returns {Promise<string|null>}
 */
async function completeWithFallback(messages, opts = {}) {
  const budgets = [
    opts.maxTokens || 1024,
    512,
    256,
  ];

  for (const budget of budgets) {
    const text = await complete(messages, { ...opts, maxTokens: budget, retry: false });
    if (text !== null) return text;
  }
  return null;
}

/**
 * Streaming chat completion.
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {object}   [opts]
 * @param {function} onChunk
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
          const lines = buffer.split('\n');
          buffer = lines.pop();
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

// ── Structured output parsing ─────────────────────────────────────────────────

/**
 * Extract named sections from a free-text LLM response.
 * Handles both "SECTION:\n- items" and "**SECTION**:\n- items" formats.
 *
 * @param {string}   text       — raw LLM output
 * @param {string[]} sections   — section names to extract
 * @returns {object}  { [sectionName]: string[] }
 */
function parseStructured(text, sections) {
  const result = {};
  if (!sections || !sections.length) return result;
  // Always initialise all requested sections to [] so callers can rely on the shape
  for (const s of sections) result[s] = [];
  if (!text) return result;

  for (const section of sections) {
    // Match "SECTION NAME:", "**SECTION NAME**:", or "### SECTION NAME"
    const pattern = new RegExp(
      `(?:\\*{1,2}|#{1,3}\\s*)?${section.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\*{0,2}:?\\s*\\n([\\s\\S]*?)(?=\\n(?:\\*{1,2}|#{1,3}\\s*)?[A-Z][A-Z ]{2,}\\s*\\*{0,2}:?|$)`,
      'i'
    );
    const match = pattern.exec(text);
    if (!match) { result[section] = []; continue; }

    // Parse bullet lines
    const block = match[1] || '';
    const items = block
      .split('\n')
      .map(l => l.replace(/^[\s\-\*\•\–]+/, '').trim())
      .filter(l => l.length > 0);
    result[section] = items;
  }

  return result;
}

// ── High-quality prompt builders ──────────────────────────────────────────────

/**
 * Build a high-quality AI match prompt for scoring a practitioner against a role.
 *
 * Context fields:
 *   practitionerName, practitionerBand, practitionerRole, practitionerSkills,
 *   practitionerCerts, practitionerSecondaryJRS, rspComments,
 *   seatTitle, seatClient, seatProject, seatBandLow, seatBandHigh,
 *   seatRequiredSkills, seatNiceSkills, seatLocation, seatStartDate,
 *   keywordScore, semanticScore
 *
 * @param {object} context
 * @returns {Array<{role:string, content:string}>}
 */
function buildMatchPrompt(context) {
  const c = context || {};

  const practSkills  = [].concat(c.practitionerSkills  || []).join(', ') || 'Not specified';
  const practCerts   = [].concat(c.practitionerCerts   || []).join(', ') || 'None';
  const reqSkills    = [].concat(c.seatRequiredSkills  || []).join(', ') || 'Not specified';
  const niceSkills   = [].concat(c.seatNiceSkills      || []).join(', ') || 'None';
  const kwScore      = c.keywordScore  != null ? `${c.keywordScore}/100` : 'N/A';
  const smScore      = c.semanticScore != null ? `${c.semanticScore}/100` : 'N/A';

  const system = `You are a precise talent-matching analyst. Your job is to assess whether a practitioner is a good fit for a specific role opening.

Output EXACTLY these four sections, in this order, using these exact headings:
OVERALL MATCH SCORE: [integer 0-100]
STRENGTHS:
- [specific strength point]
GAPS:
- [specific gap or missing skill]
RECOMMENDATION:
[1-2 sentence actionable recommendation for the hiring manager]

Rules:
- Be specific: name actual skills, technologies, and experience levels.
- STRENGTHS must name exact matching skills or capabilities — no generic praise.
- GAPS must name exact missing or weak areas relevant to the role — no generic caveats.
- RECOMMENDATION must include a clear action (pursue / shortlist with caveat / deprioritise).
- Do NOT add extra sections or preamble.
- Overall match score must reflect keyword score ${kwScore} and semantic score ${smScore} as data inputs.`;

  const user = `PRACTITIONER:
Name: ${c.practitionerName || 'Unknown'}
Band: ${c.practitionerBand || 'Unknown'}
Current Role: ${c.practitionerRole || 'Unknown'}
Secondary JRS: ${c.practitionerSecondaryJRS || 'None'}
Skills: ${practSkills}
Certifications: ${practCerts}
${c.rspComments ? `Profile Notes: ${c.rspComments}` : ''}

ROLE OPENING:
Title: ${c.seatTitle || 'Unknown'}
Client: ${c.seatClient || 'Unknown'}
Project: ${c.seatProject || 'Unknown'}
Band Range: ${c.seatBandLow || '?'} – ${c.seatBandHigh || '?'}
Location: ${c.seatLocation || 'Unknown'}
Start Date: ${c.seatStartDate || 'Unknown'}
Required Skills: ${reqSkills}
Nice-to-Have Skills: ${niceSkills}

COMPUTED SCORES:
Keyword Match Score: ${kwScore}
Semantic Match Score: ${smScore}

Provide your assessment using the exact output format specified.`;

  return [
    { role: 'system', content: system },
    { role: 'user',   content: user   },
  ];
}

/**
 * Build an assessment prompt for a CV against a job role (QCR use case).
 *
 * Context fields:
 *   cvExcerpt, jobRole, suggestedBand, keywordScore, semanticScore,
 *   keywordMatched, keywordGaps, demandCount
 *
 * @param {object} context
 * @returns {Array<{role:string, content:string}>}
 */
function buildAssessPrompt(context) {
  const c = context || {};

  const matched  = [].concat(c.keywordMatched || []).join(', ') || 'None identified';
  const gaps     = [].concat(c.keywordGaps    || []).join(', ') || 'None identified';
  const kwScore  = c.keywordScore  != null ? `${c.keywordScore}/100` : 'N/A';
  const smScore  = c.semanticScore != null ? `${c.semanticScore}/100` : 'N/A';

  const system = `You are a senior career development analyst specialising in IBM practitioner profiles.

Given a CV excerpt and scoring data, produce a focused assessment in EXACTLY this format:

STRENGTHS:
- [specific, evidence-based strength from the CV]
GAPS:
- [specific skill or experience gap relevant to the suggested role]
RECOMMENDATION:
[2-3 sentences: suitability verdict, priority development actions, and suggested next steps]

Rules:
- Every STRENGTHS bullet must cite specific technology, methodology, or achievement from the CV.
- Every GAPS bullet must name a specific missing skill or experience area for the role.
- RECOMMENDATION must be actionable — name specific technologies to learn or certifications to pursue.
- Do NOT add headers, preamble, or commentary outside the three sections.`;

  const user = `JOB ROLE: ${c.jobRole || 'Unknown'}
SUGGESTED BAND: ${c.suggestedBand || 'Unknown'}
KEYWORD SCORE: ${kwScore} (matched: ${matched})
KEYWORD GAPS: ${gaps}
SEMANTIC SCORE: ${smScore}
OPEN DEMAND SEATS FOR THIS ROLE: ${c.demandCount != null ? c.demandCount : 'Unknown'}

CV EXCERPT (first 2500 chars):
${(c.cvExcerpt || '').slice(0, 2500)}

Provide your assessment in the exact format specified.`;

  return [
    { role: 'system', content: system },
    { role: 'user',   content: user   },
  ];
}

module.exports = {
  complete,
  completeWithFallback,
  stream,
  isAvailable,
  parseStructured,
  buildMatchPrompt,
  buildAssessPrompt,
  _setConfig,
};
