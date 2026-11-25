/**
 * Loads the files written by the Python pipeline (see docs/data-format.md)
 * into typed arrays the engine can scan quickly.
 */

export interface AxisMeta {
  key: string;
  label: string;
  negative: string;
  positive: string;
  description: string;
  loo_accuracy: number;
}

export interface Meta {
  model: string;
  backend: string;
  stand_in: boolean;
  count: number;
  dims: number;
  scale: number[];
  shards: string[];
  axes: AxisMeta[];
  built: string;
}

export interface Lexicon {
  text: string[];
  group: Int32Array;            // inflection group id (shared by "difficulty" / "difficulties")
  zipf: Float32Array;
  phrase: Uint8Array;
  offensive: Uint8Array;
  phones: string[];
  syllables: Uint8Array;
  stress: string[];
  soft: Float32Array;           // z-scored softness; 0 when unknown
  hasSound: Uint8Array;
  first: string[];
  rhyme: string[];
  guessed: Uint8Array;          // pronunciation guessed by a model, not from the CMU dictionary
  axes: Record<string, Float32Array>;
}

export interface Space {
  meta: Meta;
  count: number;
  dims: number;
  vectors: Int8Array;           // count × dims, row-major
  dequant: Float32Array;        // per-dimension scale / 127
  index: Map<string, number>;
  lex: Lexicon;
  row(r: number): Float32Array; // dequantised, unit length
}

export interface RawLexicon {
  text: string[]; lemma: string[]; zipf: number[]; phrase: number[]; offensive: number[];
  phones: string[]; syllables: number[]; stress: string[]; soft: (number | null)[];
  first: string[]; rhyme: string[]; guessed: number[]; axes: Record<string, (number | null)[]>;
}

async function fetchWithProgress(url: string, onBytes: (n: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not load ${url} (${res.status})`);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

export async function loadSpace(base: string, onProgress: (loadedMB: number) => void): Promise<Space> {
  let loaded = 0;
  const tick = (n: number) => { loaded += n; onProgress(loaded / 1e6); };
  const meta: Meta = await (await fetch(`${base}/meta.json`)).json();
  const [lexBuf, ...shards] = await Promise.all([
    fetchWithProgress(`${base}/lexicon.json`, tick),
    ...meta.shards.map((s) => fetchWithProgress(`${base}/${s}`, tick)),
  ]);
  const raw: RawLexicon = JSON.parse(new TextDecoder().decode(lexBuf));
  return assembleSpace(meta, raw, shards);
}

/** Turn the parsed files into a Space. Separate from fetching so tests can call it. */
export function assembleSpace(meta: Meta, raw: RawLexicon, shards: ArrayBuffer[]): Space {
  const vectors = new Int8Array(meta.count * meta.dims);
  let off = 0;
  for (const [i, buf] of shards.entries()) {
    // Shards are raw int8 (.bin), or base64 text (.txt) for hosts that won't serve binary files
    const bytes = meta.shards[i].endsWith(".txt") ? base64ToBytes(new TextDecoder().decode(buf)) : new Uint8Array(buf);
    vectors.set(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), off);
    off += bytes.byteLength;
  }
  const dequant = Float32Array.from(meta.scale, (s) => s / 127);

  const groups = new Map<string, number>();
  const group = Int32Array.from(raw.text, (t, i) => {
    const key = raw.lemma[i] || t;
    if (!groups.has(key)) groups.set(key, groups.size);
    return groups.get(key)!;
  });
  const axes: Record<string, Float32Array> = {};
  for (const [k, v] of Object.entries(raw.axes)) axes[k] = Float32Array.from(v, (x) => x ?? 0);

  const lex: Lexicon = {
    text: raw.text,
    group,
    zipf: Float32Array.from(raw.zipf),
    phrase: Uint8Array.from(raw.phrase),
    offensive: Uint8Array.from(raw.offensive),
    phones: raw.phones,
    syllables: Uint8Array.from(raw.syllables),
    stress: raw.stress,
    soft: Float32Array.from(raw.soft, (x) => x ?? 0),
    hasSound: Uint8Array.from(raw.soft, (x) => (x === null ? 0 : 1)),
    first: raw.first,
    rhyme: raw.rhyme,
    guessed: Uint8Array.from(raw.guessed),
    axes,
  };

  const { dims } = meta;
  return {
    meta,
    count: meta.count,
    dims,
    vectors,
    dequant,
    index: new Map(raw.text.map((t, i) => [t, i])),
    lex,
    row(r: number) {
      const v = new Float32Array(dims);
      let n = 0;
      for (let d = 0; d < dims; d++) { v[d] = vectors[r * dims + d] * dequant[d]; n += v[d] * v[d]; }
      n = Math.sqrt(n) || 1;
      for (let d = 0; d < dims; d++) v[d] /= n;
      return v;
    },
  };
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64.trim());
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Prefix search for the pot input: common words first. */
export function suggest(space: Space, prefix: string, limit = 7): string[] {
  const p = prefix.trim().toLowerCase();
  if (!p) return [];
  const hits: number[] = [];
  const { text, zipf, offensive } = space.lex;
  for (let r = 0; r < space.count; r++) {
    if (!offensive[r] && text[r].startsWith(p)) hits.push(r);
  }
  hits.sort((a, b) => zipf[b] - zipf[a] || text[a].length - text[b].length);
  return hits.slice(0, limit).map((r) => text[r]);
}
