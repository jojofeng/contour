/**
 * The scoring engine: pure functions, no DOM. docs/how-it-works.md has the
 * formulas; pipeline/contour_pipeline/space.py mirrors them in Python.
 *
 *   score(c) = fit(c) − avoid(c) + dials(c) + sound(c) − rarity(c)
 *
 * Ranking has two stages, so the dials can't drag in words that don't fit:
 *   1. meaning = fit − avoid; keep the SHORTLIST best-fitting entries
 *   2. add dials + sound − rarity and re-rank that shortlist
 * (With an empty pot, the dials rank the whole vocabulary.)
 *
 *   z_i(c)   similarity of candidate c to pot item i, standardised over the
 *            whole vocabulary ("how many standard deviations closer than a
 *            typical word")
 *   fit      soft-AND of z_i over positive pot items:
 *            −(1/β)·ln Σ ŵ_i·exp(−β·z_i); β = 0 is the weighted mean (blend),
 *            large β approaches the minimum (must fit every item)
 *   avoid    Σ |w_j|·max(0, z_j) over negative pot items
 *   dials    Σ g_a·axis_a(c), with g_a ∈ [−1, 1] and axis scores already z-scored
 *   sound    g_soft·softness(c) (z-scored); 0 when the pronunciation is unknown
 *   rarity   r·max(0, 3.5 − zipf(c))/2: rare words lose up to 1.75 σ at r = 1
 */

import type { Space } from "./data";

export interface PotItem {
  id: string;
  text: string;
  weight: number;          // −2 … 2; negative means "avoid this sense"
  vector: Float32Array;    // unit length
  approx: boolean;         // true when composed from the item's words
  parts: string[];         // lexicon words it was composed from (approx only)
}

export interface Settings {
  beta: number;                      // 0 … 4
  dials: Record<string, number>;     // −1 … 1
  softness: number;                  // −1 … 1
  syllables: number;                 // 0 = any
  startsLike: string;                // a word; candidates must share its first sound
  rhymesWith: string;                // a word; candidates must share its rhyme
  rarity: number;                    // 0 … 1
  phrases: boolean;
  hideOffensive: boolean;
}

export interface Breakdown {
  fit: number;
  perItem: { text: string; weight: number; z: number }[];
  avoid: number;
  dials: { key: string; gain: number; value: number }[];
  sound: number;
  rarity: number;
  total: number;
}

export interface Result {
  index: number;
  score: number;
  breakdown: Breakdown;
  z: number[];             // per pot item, same order as the pot
}

export const DEFAULT_BETA = 2;
export const SHORTLIST = 500;
const RARITY_PIVOT = 3.5;

/** Cosine similarity of every vocabulary entry to `q`, standardised to z-scores. */
export function zSimilarities(space: Space, q: Float32Array): Float32Array {
  const { count, dims, vectors, dequant } = space;
  const qs = new Float32Array(dims);
  for (let d = 0; d < dims; d++) qs[d] = q[d] * dequant[d];
  const out = new Float32Array(count);
  let sum = 0;
  let sumSq = 0;
  for (let r = 0, base = 0; r < count; r++, base += dims) {
    let s = 0;
    for (let d = 0; d < dims; d++) s += vectors[base + d] * qs[d];
    out[r] = s;
    sum += s;
    sumSq += s * s;
  }
  const mean = sum / count;
  const sd = Math.sqrt(Math.max(sumSq / count - mean * mean, 1e-12));
  for (let r = 0; r < count; r++) out[r] = (out[r] - mean) / sd;
  return out;
}

/** Soft-AND (log-sum-exp soft minimum) of per-item z-scores for one candidate. */
export function softAnd(z: number[], weights: number[], beta: number): number {
  const total = weights.reduce((a, b) => a + b, 0);
  if (beta <= 1e-6) return z.reduce((acc, zi, i) => acc + (weights[i] / total) * zi, 0);
  const m = Math.min(...z);
  let s = 0;
  for (let i = 0; i < z.length; i++) s += (weights[i] / total) * Math.exp(-beta * (z[i] - m));
  return m - Math.log(s) / beta;
}

/** Smoothing constant for SIF weights (Arora, Liang & Ma 2017): weight = a / (a + p(word)). */
export const SIF_A = 1e-3;

/**
 * Unit vector for a pot item. If the whole text is in the vocabulary, its own
 * row. Otherwise the text is split into the longest vocabulary units
 * (multiword expressions first, left to right) and their vectors are averaged
 * with smooth-inverse-frequency weights, so "of" counts far less than "running".
 */
export function vectorFor(space: Space, text: string): { vector: Float32Array; approx: boolean; parts: string[] } | null {
  const key = text.trim().toLowerCase().replace(/\s+/g, " ");
  const row = space.index.get(key);
  if (row !== undefined) return { vector: space.row(row), approx: false, parts: [] };
  const words = key.split(" ");
  const parts: string[] = [];
  for (let i = 0; i < words.length; ) {
    let len = Math.min(4, words.length - i);
    while (len > 0 && !space.index.has(words.slice(i, i + len).join(" "))) len--;
    if (len === 0) { i++; continue; }  // unknown word: skip it
    parts.push(words.slice(i, i + len).join(" "));
    i += len;
  }
  if (!parts.length) return null;
  const v = new Float32Array(space.dims);
  for (const unit of parts) {
    const r = space.index.get(unit)!;
    const w = SIF_A / (SIF_A + 10 ** (space.lex.zipf[r] - 9));
    const vec = space.row(r);
    for (let d = 0; d < space.dims; d++) v[d] += w * vec[d];
  }
  return { vector: normalise(v), approx: true, parts };
}

export function normalise(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

/** Vocabulary rows that may be shown, given the pot and the filters. */
function allowedMask(space: Space, pot: PotItem[], s: Settings): Uint8Array {
  const { lex } = space;
  const mask = new Uint8Array(space.count).fill(1);
  const banned = new Set<number>();
  for (const item of pot) {
    for (const t of [item.text, ...item.parts]) {
      const r = space.index.get(t.toLowerCase());
      if (r !== undefined) banned.add(lex.group[r]);
    }
  }
  const first = s.startsLike ? firstSound(space, s.startsLike) : null;
  const rhyme = s.rhymesWith ? rhymeOf(space, s.rhymesWith) : null;
  for (let r = 0; r < space.count; r++) {
    if (
      banned.has(lex.group[r]) ||
      (!s.phrases && lex.phrase[r]) ||
      (s.hideOffensive && lex.offensive[r]) ||
      (s.syllables && lex.syllables[r] !== s.syllables) ||
      (first !== null && lex.first[r] !== first) ||
      (rhyme !== null && (lex.rhyme[r] !== rhyme || lex.text[r] === s.rhymesWith))
    ) {
      mask[r] = 0;
    }
  }
  return mask;
}

export function firstSound(space: Space, word: string): string {
  const r = space.index.get(word.toLowerCase());
  return r === undefined ? word.trim().toUpperCase() : space.lex.first[r];
}

export function rhymeOf(space: Space, word: string): string | null {
  const r = space.index.get(word.toLowerCase());
  return r === undefined ? null : space.lex.rhyme[r];
}

/**
 * Rank the vocabulary. `zByItem[i]` are the cached z-similarities for pot[i].
 * Returns the best `limit` results, one per inflection group.
 */
export function rank(space: Space, pot: PotItem[], zByItem: Float32Array[], s: Settings, limit = 40): Result[] {
  const pos = pot.map((p, i) => [p, i] as const).filter(([p]) => p.weight > 0);
  const neg = pot.map((p, i) => [p, i] as const).filter(([p]) => p.weight < 0);
  if (!pos.length && !Object.values(s.dials).some(Boolean) && !s.softness) return [];

  const { lex } = space;
  const mask = allowedMask(space, pot, s);
  const dialKeys = Object.keys(s.dials).filter((k) => s.dials[k]);
  const posW = pos.map(([p]) => p.weight);
  const zBuf = new Array<number>(pos.length);

  const meaning = (r: number): number => {
    let total = 0;
    if (pos.length) {
      for (let i = 0; i < pos.length; i++) zBuf[i] = zByItem[pos[i][1]][r];
      total += softAnd(zBuf, posW, s.beta);
    }
    for (const [p, i] of neg) total -= -p.weight * Math.max(0, zByItem[i][r]);
    return total;
  };
  const modifiers = (r: number): number => {
    let total = s.softness * lex.soft[r];
    for (const k of dialKeys) total += s.dials[k] * lex.axes[k][r];
    return total - (s.rarity * Math.max(0, RARITY_PIVOT - lex.zipf[r])) / 2;
  };

  // Stage 1: shortlist by meaning (or by modifiers alone when the pot is empty)
  const potEmpty = !pot.length;
  const shortlist = topRows(space.count, mask, potEmpty ? modifiers : meaning, potEmpty ? limit * 8 : SHORTLIST);
  // Stage 2: re-rank the shortlist with the modifiers
  const final = shortlist.map((r) => ({ r, v: potEmpty ? modifiers(r) : meaning(r) + modifiers(r) }));
  final.sort((a, b) => b.v - a.v || a.r - b.r);
  const order = final.map((x) => x.r);

  const seen = new Set<number>();
  const results: Result[] = [];
  for (const r of order) {
    if (seen.has(lex.group[r])) continue;
    seen.add(lex.group[r]);
    results.push(explain(space, pot, zByItem, s, r));
    if (results.length === limit) break;
  }
  return results;
}

/** Indices of the `k` highest values of `value(r)` among rows where mask[r] = 1, best first. */
export function topRows(count: number, mask: Uint8Array, value: (r: number) => number, k: number): number[] {
  const heapIdx = new Int32Array(k);
  const heapVal = new Float64Array(k);
  let size = 0;
  const swap = (a: number, b: number) => {
    [heapVal[a], heapVal[b]] = [heapVal[b], heapVal[a]];
    [heapIdx[a], heapIdx[b]] = [heapIdx[b], heapIdx[a]];
  };
  for (let r = 0; r < count; r++) {
    if (!mask[r]) continue;
    const v = value(r);
    if (size < k) {
      let i = size++;
      heapVal[i] = v; heapIdx[i] = r;
      while (i > 0 && heapVal[(i - 1) >> 1] > heapVal[i]) { swap(i, (i - 1) >> 1); i = (i - 1) >> 1; }
    } else if (v > heapVal[0]) {
      heapVal[0] = v; heapIdx[0] = r;
      for (let i = 0; ;) {
        const l = 2 * i + 1, rr = l + 1;
        let m = i;
        if (l < size && heapVal[l] < heapVal[m]) m = l;
        if (rr < size && heapVal[rr] < heapVal[m]) m = rr;
        if (m === i) break;
        swap(i, m); i = m;
      }
    }
  }
  return Array.from({ length: size }, (_, i) => i)
    .sort((a, b) => heapVal[b] - heapVal[a] || heapIdx[a] - heapIdx[b])
    .map((i) => heapIdx[i]);
}

/** Itemised score for one candidate: the numbers behind the ranking. */
export function explain(space: Space, pot: PotItem[], zByItem: Float32Array[], s: Settings, r: number): Result {
  const { lex } = space;
  const z = pot.map((_, i) => zByItem[i][r]);
  const posIdx = pot.map((p, i) => (p.weight > 0 ? i : -1)).filter((i) => i >= 0);
  const fit = posIdx.length ? softAnd(posIdx.map((i) => z[i]), posIdx.map((i) => pot[i].weight), s.beta) : 0;
  const avoid = pot.reduce((acc, p, i) => acc + (p.weight < 0 ? -p.weight * Math.max(0, z[i]) : 0), 0);
  const dials = Object.keys(s.dials)
    .filter((k) => s.dials[k])
    .map((k) => ({ key: k, gain: s.dials[k], value: s.dials[k] * lex.axes[k][r] }));
  const sound = s.softness * lex.soft[r];
  const rarity = (s.rarity * Math.max(0, RARITY_PIVOT - lex.zipf[r])) / 2;
  const total = fit - avoid + dials.reduce((a, d) => a + d.value, 0) + sound - rarity;
  return {
    index: r,
    score: total,
    z,
    breakdown: {
      fit,
      perItem: pot.map((p, i) => ({ text: p.text, weight: p.weight, z: z[i] })),
      avoid, dials, sound, rarity, total,
    },
  };
}
