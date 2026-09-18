# @futurenow/inference

Zero-dependency, CommonJS Node.js module that wraps all local AI capability
for FutureNow tools behind a single clean programmatic API.

- **Hardware detection** — CPU / CUDA GPU / Intel NPU / Apple Silicon
- **Model catalogue** — 5 GGUF models with hardware-aware recommendation
- **Sidecar lifecycle** — spawn / stop llama-cpp-python (LLM) + OpenVINO (embedding) sidecars; reuse if already running
- **Text embedding** — single and batch via OpenVINO sidecar (gte-small, 384-dim)
- **Keyword scoring** — 6-algorithm composite (Jaro-Winkler, Jaccard, Dice, Levenshtein, Damerau-Levenshtein, Cosine) with positional skill-set weighting
- **Semantic scoring** — cosine similarity, sigmoid normalisation, alpha/beta composite blend
- **LLM generation** — streaming and non-streaming via OpenAI-compatible local sidecar
- **3-tier vector cache** — hot `Map` (LRU) + TTL `Map` + optional SQLite cold store

No build step. No TypeScript. Runs in any FutureNow Express app with a single `require`.

---

## Installation

### Development (npm link)

```bash
# In this repo
cd futurenow-inference
npm link

# In the consuming project
npm link @futurenow/inference
```

### File reference

```json
"dependencies": {
  "@futurenow/inference": "file:../futurenow-inference"
}
```

### Optional peer dependency

Install `better-sqlite3` only if you want persistent cold-store vector caching:

```bash
npm install better-sqlite3
```

If `better-sqlite3` is not installed, the cold tier is silently skipped — the module
still works fully using the hot and TTL tiers.

---

## Quick start

```js
const inference = require('@futurenow/inference');

inference.init({
  modelDir:      '/path/to/models',         // required — directory with .gguf files
  embedPort:     52001,                     // default
  llmPort:       52000,                     // default
  coldStorePath: '/path/to/vectors.db',     // optional — enables SQLite cold tier
  pythonBin:     'python',                  // default
});
```

`init()` must be called once before using `sidecar`, `embed`, `llm`, or `cache`.
`hardware`, `keyword`, and `semantic` are usable immediately without `init()`.

---

## API reference

### `inference.init(opts)`

Initialise the module. Must be called before using any I/O namespaces.

| Option | Type | Default | Description |
|---|---|---|---|
| `modelDir` | `string` | **required** | Directory containing `.gguf` model files |
| `embedPort` | `number` | `52001` | OpenVINO embed sidecar port |
| `llmPort` | `number` | `52000` | llama-cpp-python LLM sidecar port |
| `coldStorePath` | `string` | `null` | SQLite path for cold vector cache |
| `pythonBin` | `string` | `'python'` | Python binary for spawning sidecars |
| `embedScript` | `string` | `null` | Override path to `embed_server.py` |
| `llmScript` | `string` | `null` | Override path to `llm_server.py` |

---

### `inference.hardware`

```js
const info     = inference.hardware.detect();       // sync, cached
const provider = inference.hardware.getProvider();  // 'webgpu' | 'cpu'
inference.hardware._resetCache();                   // force re-detect
```

**`detect()` returns:**

```js
{
  cpu:              true,              // always true
  gpu:              boolean,
  npu:              boolean,
  cuda:             boolean,
  dml:              boolean,           // DirectML (Windows)
  appleSilicon:     boolean,
  intelNpu:         boolean,
  platform:         string,            // process.platform
  arch:             string,            // process.arch
  cpuModel:         string,            // first 60 chars of CPU model string
  selectedProvider: 'webgpu' | 'cpu',
}
```

---

### `inference.models`

```js
const all        = inference.models.all();          // full catalogue array
const active     = inference.models.getActive();    // currently selected model config
inference.models.select('qwen2.5-7b-instruct');     // set active model by id
const recommended = inference.models.recommend();   // ids suited to current hardware
```

Available model IDs: `qwen2.5-7b-instruct`, `llama3.1-8b-instruct`, `mistral-7b-instruct`,
`llama3.2-3b-instruct`, `qwen2.5-14b-instruct`.

---

### `inference.sidecar`

Sidecar processes are shared with SupplyMatch when both are running. The module probes
ports first; if a sidecar is already listening it reuses it rather than spawning again.

```js
await inference.sidecar.startAll();        // start both llm + embed sidecars
await inference.sidecar.stopAll();         // graceful SIGTERM both

await inference.sidecar.startLlm();        // start llm sidecar only
await inference.sidecar.startEmbed();      // start embed sidecar only

const ok = await inference.sidecar.probe(52001, 2000);  // TCP probe, 2s timeout
```

---

### `inference.embed`

Requires the OpenVINO embed sidecar to be running (port 52001 by default).
Returns `null` on any sidecar failure — never throws.

```js
const vec    = await inference.embed.single('Senior React Developer');
// → Float32Array(384) | null

const vecs   = await inference.embed.batch(['React', 'TypeScript', 'Node.js']);
// → Float32Array[] | null
```

---

### `inference.keyword`

Pure functions — no I/O, no `init()` required.

```js
// Score a practitioner's skills against required + nice-to-have lists
const result = inference.keyword.score(
  ['react', 'typescript'],   // practitioner skills
  ['react', 'node'],         // required skills
  ['typescript']             // nice-to-have skills
);
// → { score: 0–100, tiers: number[], matched: string[], gaps: string[] }

// Low-level: composite string similarity score
const { composite, breakdown } = inference.keyword.computeKeywordScore('react', 'reactjs');
// → { composite: 0–100, breakdown: [...algorithm detail objects] }
```

---

### `inference.semantic`

Pure math — no I/O, no `init()` required.

```js
const cos       = inference.semantic.cosine(vecA, vecB);
// → number in [-1, 1]

const score     = inference.semantic.normalize(cos);
// → integer 0–100 (sigmoid, centre 0.70, scale 10)

const score2    = inference.semantic.normalize(cos, {
  centre: 0.65, scale: 8, asPercent: false
});
// → float 0–1

const blended   = inference.semantic.composite(smScore, kwScore, 0.40, 0.60);
// → integer 0–100 (α × semantic + β × keyword)
```

---

### `inference.llm`

Requires the llama-cpp-python sidecar to be running (port 52000 by default).
Returns `null` on any failure — never throws.

```js
// Check if sidecar is available
const alive = await inference.llm.isAvailable();
// → boolean

// Non-streaming completion
const text = await inference.llm.complete([
  { role: 'system',  content: 'You are a helpful assistant.' },
  { role: 'user',    content: 'Summarise these performance notes.' },
], {
  maxTokens:   1024,
  temperature: 0.3,
  topP:        0.9,
  model:       'local-model',   // sidecar ignores this; included for API compat
  timeoutMs:   120_000,
});
// → string | null

// Streaming completion — calls onChunk() for every token delta
await inference.llm.stream(
  messages,
  { maxTokens: 2048 },
  (delta) => process.stdout.write(delta)
);
```

---

### `inference.cache`

3-tier vector cache. Requires `init()`.

```js
// Store
inference.cache.set('supply:123', vec, 'hot');   // hot tier only (default)
inference.cache.set('supply:123', vec, 'ttl');   // hot + TTL (expires after ttlMs)
inference.cache.set('supply:123', vec, 'cold');  // all three tiers + SQLite

// Retrieve (hot → ttl → cold, promotes on hit)
const vec = inference.cache.get('supply:123');
// → Float32Array | null

// Remove from all tiers
inference.cache.del('supply:123');

// Clear in-process tiers (does NOT wipe SQLite cold store)
inference.cache.flush();

// Entry counts
const { hot, ttl, cold } = inference.cache.stats();
```

**Config options** (passed through `init()`):

| Option | Default | Description |
|---|---|---|
| `coldStorePath` | `null` | SQLite file path; omit to disable cold tier |
| `hotMax` | `500` | Maximum hot-tier entries before LRU eviction |
| `ttlMs` | `600000` | TTL tier expiry in milliseconds (default 10 min) |

---

## Running tests

```bash
node test/run-all.js
```

All tests are pure unit tests — no network, no sidecars, no GPU required.
The test suite covers `hardware`, `keyword`, `semantic`, and `cache`.

---

## Architecture notes

- **CommonJS only** — zero build step, consistent with SupplyMatch conventions.
- **localhost binding** — all sidecar communication is `127.0.0.1` only; never `0.0.0.0`.
- **No SupplyMatch imports** — this module is fully self-contained. SupplyMatch source was
  used as reference only; all logic is re-implemented from scratch.
- **Sidecar reuse** — if SupplyMatch (or another FutureNow tool) has already started the
  llm/embed sidecars, `sidecar.startAll()` detects the live ports and skips spawning.
- **Graceful nulls** — `embed.single()`, `embed.batch()`, `llm.complete()`, and `cache.get()`
  all return `null` on failure rather than throwing; callers should always null-check.

---

## Licence

Internal FutureNow tooling. Not for public distribution.
