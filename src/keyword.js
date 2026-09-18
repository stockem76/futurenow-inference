'use strict';
/**
 * keyword.js — Keyword scoring for @futurenow/inference.
 *
 * Adapted from SupplyMatch keywordScore.js + stringMetrics.js.
 * Pure functions only — no I/O, no SupplyMatch imports.
 *
 * Exports:
 *   score(practSkills, reqSkills, niceSkills, opts)
 *     → { score: 0-100, tiers: number[], matched: string[], gaps: string[] }
 *   computeKeywordScore(a, b, weights?)
 *     → { composite: 0-100, breakdown: Array }
 *   DEFAULT_WEIGHTS
 */

// ── String metrics (inlined — no SupplyMatch import) ─────────────────────────

function levenshtein(a, b) {
  a = String(a||''); b = String(b||'');
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const curr = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i-1] === b[j-1] ? 0 : 1;
      curr[j] = Math.min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost);
    }
    prev = curr;
  }
  return prev[b.length];
}

function damerauLevenshtein(a, b) {
  a = String(a||''); b = String(b||'');
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const m = a.length, n = b.length;
  const d = Array.from({ length: m+2 }, () => new Array(n+2).fill(0));
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
  return d[m+1][n+1];
}

function jaroWinkler(a, b, p = 0.1) {
  a = String(a||'').toLowerCase(); b = String(b||'').toLowerCase();
  if (!a.length && !b.length) return 0;
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  const mw = Math.max(0, Math.floor(Math.max(a.length, b.length) / 2) - 1);
  const am = new Array(a.length).fill(false);
  const bm = new Array(b.length).fill(false);
  let matches = 0, trans = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = Math.max(0, i-mw); j <= Math.min(b.length-1, i+mw); j++) {
      if (bm[j] || a[i] !== b[j]) continue;
      am[i] = true; bm[j] = true; matches++; break;
    }
  }
  if (!matches) return 0;
  let k = 0;
  for (let i = 0; i < a.length; i++) {
    if (!am[i]) continue; while (!bm[k]) k++;
    if (a[i] !== b[k]) trans++; k++;
  }
  const jaro = (matches/a.length + matches/b.length + (matches - trans/2)/matches) / 3;
  let prefix = 0;
  const mp = Math.min(4, Math.min(a.length, b.length));
  while (prefix < mp && a[prefix] === b[prefix]) prefix++;
  return jaro + prefix * p * (1 - jaro);
}

function jaccard(a, b) {
  a = String(a||'').toLowerCase(); b = String(b||'').toLowerCase();
  const tokenise = s => new Set(s.split(/\W+/).filter(Boolean));
  const sa = tokenise(a); const sb = tokenise(b);
  if (!sa.size && !sb.size) return 1;
  if (!sa.size || !sb.size) return 0;
  let intersect = 0;
  for (const t of sa) { if (sb.has(t)) intersect++; }
  return intersect / (sa.size + sb.size - intersect);
}

function sorensenDice(a, b, n = 2) {
  a = String(a||'').toLowerCase(); b = String(b||'').toLowerCase();
  if (a === b) return 1;
  const ngrams = s => { const m = new Map(); for (let i = 0; i <= s.length-n; i++) { const g = s.slice(i,i+n); m.set(g,(m.get(g)||0)+1); } return m; };
  const ga = ngrams(a); const gb = ngrams(b);
  const ta = a.length-n+1; const tb = b.length-n+1;
  if (ta <= 0 || tb <= 0) return 0;
  let ix = 0;
  for (const [g,c] of ga) { if (gb.has(g)) ix += Math.min(c, gb.get(g)); }
  return (2 * ix) / (ta + tb);
}

function cosineSimilarity(a, b) {
  a = String(a||'').toLowerCase(); b = String(b||'').toLowerCase();
  if (a === b && a.length) return 1;
  const freq = s => { const m = new Map(); for (const t of s.split(/\W+/).filter(Boolean)) m.set(t,(m.get(t)||0)+1); return m; };
  const fa = freq(a); const fb = freq(b);
  if (!fa.size || !fb.size) return 0;
  let dot = 0, na = 0, nb = 0;
  for (const [t,c] of fa) { na += c*c; if (fb.has(t)) dot += c*fb.get(t); }
  for (const [,c] of fb) nb += c*c;
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Composite keyword score ───────────────────────────────────────────────────

const DEFAULT_WEIGHTS = {
  levenshtein:        0.20,
  damerauLevenshtein: 0.15,
  jaroWinkler:        0.25,
  jaccard:            0.20,
  sorensenDice:       0.10,
  cosineSimilarity:   0.10,
};

const ALGO_META = {
  levenshtein:        { label: 'Levenshtein',         useCase: 'Spell check / edit distance' },
  damerauLevenshtein: { label: 'Damerau-Levenshtein', useCase: 'Transposition typos' },
  jaroWinkler:        { label: 'Jaro-Winkler',        useCase: 'Name / prefix matching' },
  jaccard:            { label: 'Jaccard',             useCase: 'Skill-set token overlap' },
  sorensenDice:       { label: 'Sørensen–Dice',       useCase: 'Bigram tag similarity' },
  cosineSimilarity:   { label: 'Cosine Similarity',   useCase: 'Document topic overlap' },
};

function normaliseWeights(weights) {
  const total = Object.values(weights).reduce((s,v) => s + Math.max(0, v||0), 0);
  if (total === 0) {
    const keys = Object.keys(weights); const eq = 1/keys.length;
    return Object.fromEntries(keys.map(k => [k, eq]));
  }
  return Object.fromEntries(Object.entries(weights).map(([k,v]) => [k, Math.max(0,v||0)/total]));
}

/**
 * Compute a composite keyword score for two strings.
 * @param {string} a
 * @param {string} b
 * @param {object} [weights]
 * @returns {{ composite: number, breakdown: Array }}
 */
function computeKeywordScore(a, b, weights) {
  const ew  = normaliseWeights(weights || DEFAULT_WEIGHTS);
  const maxLen = Math.max((a||'').length, (b||'').length) || 1;
  const raw = {
    levenshtein:        1 - levenshtein(a,b)         / maxLen,
    damerauLevenshtein: 1 - damerauLevenshtein(a,b)  / maxLen,
    jaroWinkler:        jaroWinkler(a,b),
    jaccard:            jaccard(a,b),
    sorensenDice:       sorensenDice(a,b),
    cosineSimilarity:   cosineSimilarity(a,b),
  };
  let composite = 0;
  const breakdown = [];
  for (const [algo, normWeight] of Object.entries(ew)) {
    if (!(algo in raw)) continue;
    const rawScore    = Math.max(0, Math.min(1, raw[algo]));
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

  const threshold = opts.matchThreshold || 60; // composite score ≥ threshold = "match"

  const matched = [];
  const gaps    = [];
  const tiers   = [];
  let   total   = 0;
  let   possible = 0;

  const reqLen  = reqSkills.length;
  const niceLen = niceSkills.length;

  function positionalWeight(idx, listLen, isRequired) {
    if (isRequired) {
      // 1.5 → 1.0 linearly over required list
      return listLen <= 1 ? 1.5 : 1 + 0.5 * (1 - idx / (listLen - 1));
    } else {
      // 0.75 → 0.50 linearly over nice list
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
    }
    return best;
  }

  for (let i = 0; i < reqLen; i++) {
    const skill = reqSkills[i];
    const w     = positionalWeight(i, reqLen, true);
    const s     = bestMatchScore(skill);
    const tier  = Math.round(s / 20); // 0–5 depth tier
    tiers.push(tier);
    possible += w * 100;
    if (s >= threshold) { total += w * s; matched.push(skill); }
    else                { total += 0;     gaps.push(skill); }
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

module.exports = { score, computeKeywordScore, normaliseWeights, DEFAULT_WEIGHTS, ALGO_META };
