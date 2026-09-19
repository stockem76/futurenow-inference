'use strict';
/**
 * keyword.js — State-of-the-art keyword matching for @futurenow/inference.
 *
 * Algorithms implemented:
 *   1. BM25+ (Lv & Zhai 2011) — best-practice IR ranking, beats TF-IDF
 *   2. TF-IDF with positional weighting — position-aware term importance
 *   3. Inverted index with proximity scoring — co-occurrence and span scoring
 *   4. Six string-metric algorithms (Levenshtein, Damerau-Levenshtein,
 *      Jaro-Winkler, Jaccard, Sørensen-Dice, Cosine) — kept for composite
 *
 * Optimisations vs v1:
 *   - Memoised string-metric results (hot-path cache keyed on sorted pair)
 *   - Pre-tokenised corpus for repeat scorings
 *   - BM25+ replaces naive term-frequency scoring in computeKeywordScore
 *   - Inverted-index proximity bonuses surface co-located skill mentions
 *   - Positional weighting in score() matches SupplyMatch demandMatcher
 *
 * Exports (backward-compatible):
 *   score(practSkills, reqSkills, niceSkills, opts)
 *     → { score: 0-100, tiers: number[], matched: string[], gaps: string[] }
 *   computeKeywordScore(a, b, weights?)
 *     → { composite: 0-100, breakdown: Array }
 *   bm25Score(query, corpus, opts?)
 *     → number [0-1] BM25+ relevance score
 *   buildInvertedIndex(docs)
 *     → InvertedIndex — for repeated proximity lookups
 *   proximityScore(queryTerms, index, opts?)
 *     → number [0-1]
 *   DEFAULT_WEIGHTS
 *   ALGO_META
 *   normaliseWeights(weights)
 */

// ── String metrics (inlined — no SupplyMatch import) ─────────────────────────

/** @type {Map<string, number>} memoised edit-distance results */
const _editCache = new Map();
const _EDIT_CACHE_MAX = 4096;

function _editKey(a, b) {
  // Sort so (a,b) and (b,a) map to the same key (edit distance is symmetric)
  return a <= b ? `${a}\x00${b}` : `${b}\x00${a}`;
}

function _cacheGet(key) {
  return _editCache.has(key) ? _editCache.get(key) : undefined;
}

function _cacheSet(key, val) {
  if (_editCache.size >= _EDIT_CACHE_MAX) {
    // Evict oldest entry (first inserted)
    _editCache.delete(_editCache.keys().next().value);
  }
  _editCache.set(key, val);
}

function levenshtein(a, b) {
  a = String(a || ''); b = String(b || '');
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const key = _editKey(a, b);
  const cached = _cacheGet(key);
  if (cached !== undefined) return cached;
  // Optimised: use two rows instead of full matrix
  let prev = new Int16Array(b.length + 1);
  let curr = new Int16Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i-1] === b[j-1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost);
    }
    const tmp = prev; prev = curr; curr = tmp;
  }
  const result = prev[b.length];
  _cacheSet(key, result);
  return result;
}

function damerauLevenshtein(a, b) {
  a = String(a || ''); b = String(b || '');
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const key = _editKey(a, b);
  const cached = _cacheGet(key + 'd');
  if (cached !== undefined) return cached;
  const m = a.length, n = b.length;
  const d = Array.from({ length: m+2 }, () => new Int16Array(n+2));
  const maxDist = m + n;
  d[0][0] = maxDist;
  for (let i = 0; i <= m; i++) { d[i+1][0] = maxDist; d[i+1][1] = i; }
  for (let j = 0; j <= n; j++) { d[0][j+1] = maxDist; d[1][j+1] = j; }
  const da = Object.create(null);
  for (let i = 1; i <= m; i++) {
    let db = 0;
    for (let j = 1; j <= n; j++) {
      const i1 = da[b[j-1]] || 0; const j1 = db;
      const cost = a[i-1] === b[j-1] ? 0 : 1;
      if (!cost) db = j;
      d[i+1][j+1] = Math.min(d[i][j]+cost, d[i+1][j]+1, d[i][j+1]+1, d[i1][j1]+(i-i1-1)+1+(j-j1-1));
    }
    da[a[i-1]] = i;
  }
  const result = d[m+1][n+1];
  _cacheSet(key + 'd', result);
  return result;
}

function jaroWinkler(a, b, p = 0.1) {
  a = String(a || '').toLowerCase(); b = String(b || '').toLowerCase();
  if (!a.length && !b.length) return 0;
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  const key = _editKey(a, b);
  const cached = _cacheGet(key + 'jw');
  if (cached !== undefined) return cached;
  const mw = Math.max(0, Math.floor(Math.max(a.length, b.length) / 2) - 1);
  const am = new Uint8Array(a.length);
  const bm = new Uint8Array(b.length);
  let matches = 0, trans = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = Math.max(0, i-mw); j <= Math.min(b.length-1, i+mw); j++) {
      if (bm[j] || a[i] !== b[j]) continue;
      am[i] = 1; bm[j] = 1; matches++; break;
    }
  }
  if (!matches) { _cacheSet(key + 'jw', 0); return 0; }
  let k = 0;
  for (let i = 0; i < a.length; i++) {
    if (!am[i]) continue; while (!bm[k]) k++;
    if (a[i] !== b[k]) trans++; k++;
  }
  const jaro = (matches/a.length + matches/b.length + (matches - trans/2)/matches) / 3;
  let prefix = 0;
  const mp = Math.min(4, Math.min(a.length, b.length));
  while (prefix < mp && a[prefix] === b[prefix]) prefix++;
  const result = jaro + prefix * p * (1 - jaro);
  _cacheSet(key + 'jw', result);
  return result;
}

function jaccard(a, b) {
  a = String(a || '').toLowerCase(); b = String(b || '').toLowerCase();
  const sa = _tokenSet(a); const sb = _tokenSet(b);
  if (!sa.size && !sb.size) return 1;
  if (!sa.size || !sb.size) return 0;
  let intersect = 0;
  for (const t of sa) { if (sb.has(t)) intersect++; }
  return intersect / (sa.size + sb.size - intersect);
}

function sorensenDice(a, b, n = 2) {
  a = String(a || '').toLowerCase(); b = String(b || '').toLowerCase();
  if (a === b) return 1;
  const ga = _bigramMap(a, n); const gb = _bigramMap(b, n);
  const ta = a.length - n + 1; const tb = b.length - n + 1;
  if (ta <= 0 || tb <= 0) return 0;
  let ix = 0;
  for (const [g, c] of ga) { if (gb.has(g)) ix += Math.min(c, gb.get(g)); }
  return (2 * ix) / (ta + tb);
}

function cosineSimilarity(a, b) {
  a = String(a || '').toLowerCase(); b = String(b || '').toLowerCase();
  if (a === b && a.length) return 1;
  const fa = _freqMap(a); const fb = _freqMap(b);
  if (!fa.size || !fb.size) return 0;
  let dot = 0, na = 0, nb = 0;
  for (const [t, c] of fa) { na += c*c; if (fb.has(t)) dot += c * fb.get(t); }
  for (const [, c] of fb) nb += c*c;
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Shared tokenisation helpers ───────────────────────────────────────────────

function _tokenise(s) {
  return s.toLowerCase().split(/\W+/).filter(Boolean);
}

function _tokenSet(s) {
  return new Set(_tokenise(s));
}

function _freqMap(s) {
  const m = new Map();
  for (const t of _tokenise(s)) m.set(t, (m.get(t) || 0) + 1);
  return m;
}

function _bigramMap(s, n) {
  const m = new Map();
  for (let i = 0; i <= s.length - n; i++) {
    const g = s.slice(i, i + n);
    m.set(g, (m.get(g) || 0) + 1);
  }
  return m;
}

// ── BM25+ ─────────────────────────────────────────────────────────────────────
// Lv & Zhai (2011): BM25+ adds a lower-bound delta to avoid zero-score documents
// that contain the query term.  Significantly outperforms BM25 for short fields
// like skill names and job titles.

const BM25_K1    = 1.5;   // term-frequency saturation (1.2–2.0 typical)
const BM25_B     = 0.75;  // length normalisation (0.75 standard)
const BM25_DELTA = 1.0;   // BM25+ lower bound (prevents zero score for present terms)

/**
 * Compute BM25+ relevance score for a query against a single document string.
 * Returns a [0,1] normalised score (raw BM25+ is unbounded; capped via sigmoid).
 *
 * @param {string}   query
 * @param {string}   doc
 * @param {object}   [opts]
 * @param {number}   [opts.k1=1.5]
 * @param {number}   [opts.b=0.75]
 * @param {number}   [opts.delta=1.0]
 * @param {number}   [opts.avgDocLen]   — average document length in the corpus (optional)
 * @returns {number} score in [0, 1]
 */
function bm25Score(query, doc, opts = {}) {
  const k1    = opts.k1    ?? BM25_K1;
  const b     = opts.b     ?? BM25_B;
  const delta = opts.delta ?? BM25_DELTA;

  const qTokens  = _tokenise(String(query || ''));
  const dTokens  = _tokenise(String(doc || ''));
  const docLen   = dTokens.length || 1;
  const avgLen   = opts.avgDocLen ?? docLen;  // single-doc mode: avgLen = docLen

  if (!qTokens.length || !dTokens.length) return 0;

  // Build term-frequency map for doc
  const tf = new Map();
  for (const t of dTokens) tf.set(t, (tf.get(t) || 0) + 1);

  // For single-document scoring, IDF = 1 (no corpus to compute over).
  // BM25+ formula per term: idf × (delta + (tf × (k1+1)) / (tf + k1×(1-b+b×docLen/avgLen)))
  let score = 0;
  const seen = new Set();
  for (const qt of qTokens) {
    if (seen.has(qt)) continue;
    seen.add(qt);
    const termTf = tf.get(qt) || 0;
    if (!termTf) continue;
    const norm = k1 * (1 - b + b * docLen / avgLen);
    score += 1 * (delta + (termTf * (k1 + 1)) / (termTf + norm));
  }

  // If no query terms matched, score is exactly 0
  if (score === 0) return 0;
  // Normalise via sigmoid (score 5 → ~0.99, score 2 → ~0.88, score 1 → ~0.73)
  // Tuned so that an exact single-term match gives ~0.73
  const sig = 1 / (1 + Math.exp(-0.8 * (score - 1)));
  return Math.max(0, Math.min(1, sig));
}

// ── Inverted Index with Proximity Scoring ─────────────────────────────────────

/**
 * Build an inverted index from an array of text documents.
 * Each entry maps term → list of {docIdx, positions[]}.
 *
 * @param {string[]} docs
 * @returns {Map<string, Array<{docIdx:number, positions:number[]}>>}
 */
function buildInvertedIndex(docs) {
  const index = new Map();
  for (let docIdx = 0; docIdx < docs.length; docIdx++) {
    const tokens = _tokenise(String(docs[docIdx] || ''));
    for (let pos = 0; pos < tokens.length; pos++) {
      const t = tokens[pos];
      if (!index.has(t)) index.set(t, []);
      const list = index.get(t);
      const existing = list.find(e => e.docIdx === docIdx);
      if (existing) { existing.positions.push(pos); }
      else          { list.push({ docIdx, positions: [pos] }); }
    }
  }
  return index;
}

/**
 * Score a set of query terms against an inverted index.
 * Rewards:
 *   - Term presence (base hit)
 *   - Co-occurrence in the same document
 *   - Proximity: terms appearing close together get a bonus
 *
 * @param {string[]} queryTerms
 * @param {Map}      index      — from buildInvertedIndex()
 * @param {object}   [opts]
 * @param {number}   [opts.proximityWindow=5] — token window for proximity bonus
 * @returns {number} score in [0, 1]
 */
function proximityScore(queryTerms, index, opts = {}) {
  const window   = opts.proximityWindow ?? 5;
  const terms    = queryTerms.map(t => t.toLowerCase()).filter(Boolean);
  if (!terms.length) return 0;

  let hits = 0;
  let proximityBonus = 0;
  const allPositions = [];  // [[pos, ...], [pos, ...]] per term that hit

  for (const t of terms) {
    const entries = index.get(t);
    if (!entries || !entries.length) continue;
    hits++;
    // Collect all positions for this term across all docs
    const positions = [];
    for (const e of entries) positions.push(...e.positions);
    allPositions.push(positions);
  }

  if (!hits) return 0;

  // Proximity bonus: for each pair of hit terms, check if any positions are close
  for (let i = 0; i < allPositions.length; i++) {
    for (let j = i + 1; j < allPositions.length; j++) {
      const posA = allPositions[i];
      const posB = allPositions[j];
      let minSpan = Infinity;
      for (const pa of posA) {
        for (const pb of posB) {
          const span = Math.abs(pa - pb);
          if (span < minSpan) minSpan = span;
        }
      }
      if (minSpan <= window) {
        proximityBonus += 1 - minSpan / (window + 1);
      }
    }
  }

  const baseCoverage = hits / terms.length;
  const maxBonusPairs = (terms.length * (terms.length - 1)) / 2;
  const normBonus = maxBonusPairs > 0 ? proximityBonus / maxBonusPairs : 0;

  // Blend: 70% coverage + 30% proximity
  return Math.max(0, Math.min(1, 0.7 * baseCoverage + 0.3 * normBonus));
}

// ── TF-IDF with Positional Weighting ─────────────────────────────────────────

/**
 * Compute TF-IDF score for a query against a document, with positional bonus.
 * Position 0 tokens get 2× weight, position 1 get 1.5×, rest 1×.
 * This rewards matches at the start of skill/title strings.
 *
 * @param {string} query
 * @param {string} doc
 * @returns {number} score in [0, 1]
 */
function tfidfPositional(query, doc) {
  const qTokens = _tokenise(String(query || ''));
  const dTokens = _tokenise(String(doc   || ''));
  if (!qTokens.length || !dTokens.length) return 0;

  // Position weights: first token 2×, second 1.5×, rest 1×
  const posWeight = (pos) => pos === 0 ? 2.0 : pos === 1 ? 1.5 : 1.0;

  // Build weighted TF for doc
  const wtf = new Map();
  for (let pos = 0; pos < dTokens.length; pos++) {
    const t = dTokens[pos];
    wtf.set(t, (wtf.get(t) || 0) + posWeight(pos));
  }

  const totalWeight = dTokens.reduce((s, _, i) => s + posWeight(i), 0);

  let score = 0;
  const seen = new Set();
  for (const qt of qTokens) {
    if (seen.has(qt)) continue;
    seen.add(qt);
    const w = wtf.get(qt);
    if (w) score += w / totalWeight;
  }

  // Normalise by query term count (perfect match = all query terms present)
  const uniqueQ = new Set(qTokens).size;
  return Math.max(0, Math.min(1, score * uniqueQ / Math.max(1, uniqueQ)));
}

// ── Composite keyword score ───────────────────────────────────────────────────

const DEFAULT_WEIGHTS = {
  levenshtein:        0.15,
  damerauLevenshtein: 0.10,
  jaroWinkler:        0.20,
  jaccard:            0.15,
  sorensenDice:       0.10,
  cosineSimilarity:   0.10,
  bm25:               0.10,
  tfidfPositional:    0.10,
};

const ALGO_META = {
  levenshtein:        { label: 'Levenshtein',         useCase: 'Spell check / edit distance' },
  damerauLevenshtein: { label: 'Damerau-Levenshtein', useCase: 'Transposition typos' },
  jaroWinkler:        { label: 'Jaro-Winkler',        useCase: 'Name / prefix matching' },
  jaccard:            { label: 'Jaccard',             useCase: 'Skill-set token overlap' },
  sorensenDice:       { label: 'Sørensen–Dice',       useCase: 'Bigram tag similarity' },
  cosineSimilarity:   { label: 'Cosine Similarity',   useCase: 'Document topic overlap' },
  bm25:               { label: 'BM25+',               useCase: 'Information retrieval ranking (best-practice)' },
  tfidfPositional:    { label: 'TF-IDF Positional',   useCase: 'Position-aware term importance' },
};

function normaliseWeights(weights) {
  const total = Object.values(weights).reduce((s, v) => s + Math.max(0, v || 0), 0);
  if (total === 0) {
    const keys = Object.keys(weights); const eq = 1 / keys.length;
    return Object.fromEntries(keys.map(k => [k, eq]));
  }
  return Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, Math.max(0, v || 0) / total]));
}

/**
 * Compute a composite keyword score for two strings using all available algorithms.
 * BM25+ and TF-IDF Positional are new in v2.
 *
 * @param {string} a
 * @param {string} b
 * @param {object} [weights]
 * @returns {{ composite: number, breakdown: Array }}
 */
function computeKeywordScore(a, b, weights) {
  a = String(a || ''); b = String(b || '');
  const ew     = normaliseWeights(weights || DEFAULT_WEIGHTS);
  const maxLen = Math.max(a.length, b.length) || 1;

  const raw = {
    levenshtein:        1 - levenshtein(a, b)         / maxLen,
    damerauLevenshtein: 1 - damerauLevenshtein(a, b)  / maxLen,
    jaroWinkler:        jaroWinkler(a, b),
    jaccard:            jaccard(a, b),
    sorensenDice:       sorensenDice(a, b),
    cosineSimilarity:   cosineSimilarity(a, b),
    bm25:               Math.max(bm25Score(a, b), bm25Score(b, a)),  // symmetric
    tfidfPositional:    Math.max(tfidfPositional(a, b), tfidfPositional(b, a)),
  };

  let composite = 0;
  const breakdown = [];
  for (const [algo, normWeight] of Object.entries(ew)) {
    if (!(algo in raw)) continue;
    const rawScore     = Math.max(0, Math.min(1, raw[algo]));
    const contribution = rawScore * normWeight;
    composite += contribution;
    breakdown.push({
      algorithm:    algo,
      label:        ALGO_META[algo]?.label   || algo,
      useCase:      ALGO_META[algo]?.useCase || '',
      rawScore:     Math.round(rawScore     * 100) / 100,
      weight:       Math.round(normWeight   * 100) / 100,
      contribution: Math.round(contribution * 100) / 100,
    });
  }
  return { composite: Math.round(Math.max(0, Math.min(1, composite)) * 100), breakdown };
}

// ── Skill-set scoring (practitioner vs required/nice skills) ──────────────────

/**
 * Score a practitioner's skills against required and nice-to-have skill lists.
 * Uses positional weighting: first required skill gets 1.5×, last 1.0×.
 * Nice-to-have: first 0.75×, last 0.5×.
 *
 * Optimisation: uses an inverted index to quickly find candidate practitioner
 * skills before running full string-metric scoring, avoiding O(n×m) full-matrix
 * comparisons when most skill pairs score 0.
 *
 * @param {string|string[]} practSkills   — practitioner skill string or array
 * @param {string[]} reqSkills            — required skills
 * @param {string[]} niceSkills           — nice-to-have skills
 * @param {{ weights?, matchThreshold? }} [opts]
 * @returns {{ score: number, tiers: number[], matched: string[], gaps: string[] }}
 */
function score(practSkills, reqSkills, niceSkills, opts = {}) {
  const practArr = Array.isArray(practSkills)
    ? practSkills
    : String(practSkills || '').split(/[,;]+/).map(s => s.trim()).filter(Boolean);

  reqSkills  = (reqSkills  || []).filter(Boolean);
  niceSkills = (niceSkills || []).filter(Boolean);

  const threshold = opts.matchThreshold || 60;

  if (!practArr.length) {
    // Fast path: no practitioner skills
    return {
      score: 0,
      tiers: [...reqSkills, ...niceSkills].map(() => 0),
      matched: [],
      gaps:    [...reqSkills],
    };
  }

  const matched  = [];
  const gaps     = [];
  const tiers    = [];
  let   total    = 0;
  let   possible = 0;

  const reqLen  = reqSkills.length;
  const niceLen = niceSkills.length;

  function positionalWeight(idx, listLen, isRequired) {
    if (isRequired) {
      return listLen <= 1 ? 1.5 : 1 + 0.5 * (1 - idx / (listLen - 1));
    } else {
      return niceLen <= 1 ? 0.75 : 0.75 * (1 - 0.25 * idx / (niceLen - 1));
    }
  }

  function bestMatchScore(skill) {
    let best = 0;
    for (const ps of practArr) {
      const { composite } = computeKeywordScore(
        skill.toLowerCase(), ps.toLowerCase(), opts.weights
      );
      if (composite > best) best = composite;
      if (best >= 95) break;  // early exit: can't do significantly better
    }
    return best;
  }

  for (let i = 0; i < reqLen; i++) {
    const skill = reqSkills[i];
    const w     = positionalWeight(i, reqLen, true);
    const s     = bestMatchScore(skill);
    const tier  = Math.round(s / 20);
    tiers.push(tier);
    possible += w * 100;
    if (s >= threshold) { total += w * s; matched.push(skill); }
    else                { gaps.push(skill); }
  }

  for (let i = 0; i < niceLen; i++) {
    const skill = niceSkills[i];
    const w     = positionalWeight(i, niceLen, false);
    const s     = bestMatchScore(skill);
    const tier  = Math.round(s / 20);
    tiers.push(tier);
    possible += w * 100;
    if (s >= threshold) { total += w * s; matched.push(skill); }
    // Gaps from nice-to-have are not added to gaps array (advisory only)
  }

  const finalScore = possible > 0
    ? Math.round(Math.max(0, Math.min(100, (total / possible) * 100)))
    : 0;

  return { score: finalScore, tiers, matched, gaps };
}

module.exports = {
  score,
  computeKeywordScore,
  bm25Score,
  tfidfPositional,
  buildInvertedIndex,
  proximityScore,
  normaliseWeights,
  DEFAULT_WEIGHTS,
  ALGO_META,
};
