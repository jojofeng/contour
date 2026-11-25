"""Load the exported files and score candidates with *exactly* the app's math.

`web/src/engine.ts` is the source of truth for the app; this module mirrors
it so experiments (see evaluate.py) measure what users actually see.
docs/how-it-works.md has the formulas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SIF_A = 1e-3  # smooth inverse frequency constant (Arora et al. 2017)


@dataclass
class LoadedSpace:
    meta: dict
    lex: dict
    vectors: np.ndarray          # (n, dims) float32 (dequantised)
    index: dict[str, int]
    lemma: list[str]

    @classmethod
    def load(cls, folder: str | Path) -> "LoadedSpace":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        lex = json.loads((folder / "lexicon.json").read_text())
        raw = b"".join((folder / s).read_bytes() for s in meta["shards"])
        q = np.frombuffer(raw, dtype=np.int8).reshape(meta["count"], meta["dims"])
        vectors = q.astype(np.float32) * (np.array(meta["scale"], dtype=np.float32) / 127)
        texts = lex["text"]
        lemma = [l or t for l, t in zip(lex["lemma"], texts)]
        return cls(meta, lex, vectors, {t: i for i, t in enumerate(texts)}, lemma)

    def query_vector(self, text: str) -> np.ndarray:
        """Lexicon row, or for unknown text the SIF-weighted mean of its longest
        vocabulary units (mirrors `vectorFor` in web/src/engine.ts)."""
        if text in self.index:
            v = self.vectors[self.index[text]]
            return v / np.linalg.norm(v)
        words, parts, i = text.split(), [], 0
        while i < len(words):
            n = min(4, len(words) - i)
            while n and " ".join(words[i : i + n]) not in self.index:
                n -= 1
            if n == 0:
                i += 1
                continue
            parts.append(" ".join(words[i : i + n]))
            i += n
        if not parts:
            raise KeyError(text)
        v = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for unit in parts:
            r = self.index[unit]
            r_vec = self.vectors[r] / np.linalg.norm(self.vectors[r])
            v += SIF_A / (SIF_A + 10 ** (self.lex["zipf"][r] - 9)) * r_vec
        return v / np.linalg.norm(v)

    def zsims(self, text: str) -> np.ndarray:
        s = self.vectors @ self.query_vector(text)
        return (s - s.mean()) / s.std()


def fit_score(z: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    """Soft-AND of per-item z-similarities.

    z: (k, n) standardised similarities for k positive pot items; weights: (k,) > 0.
    beta = 0 -> weighted mean ("blend"); beta -> infinity -> minimum ("must fit all").
    """
    w = weights / weights.sum()
    if beta <= 1e-6:
        return w @ z
    m = z.min(axis=0)  # stabilise the log-sum-exp
    return m - np.log(w @ np.exp(-beta * (z - m))) / beta


RARITY_PIVOT = 3.5
SHORTLIST = 500


def _col(values) -> np.ndarray:
    return np.array([x if x is not None else 0 for x in values], dtype=np.float64)


def rank_rows(space: LoadedSpace, pot: list[tuple[str, float]], beta: float, dials: dict[str, float],
              softness: float = 0.0, rarity: float = 0.0, limit: int = 10) -> list[tuple[int, float]]:
    """Mirror of `rank` in web/src/engine.ts. Returns [(row, score)], one row per lemma group.

    Stage 1 (meaning): fit - avoid over allowed rows; keep the best SHORTLIST.
    Stage 2: add dials + sound - rarity, sort, keep the best row per lemma group.
    """
    lex = space.lex
    n = len(space.vectors)
    pos = [(t, w) for t, w in pot if w > 0]
    neg = [(t, w) for t, w in pot if w < 0]
    meaning = np.zeros(n)
    if pos:
        meaning += fit_score(np.stack([space.zsims(t) for t, _ in pos]), np.array([w for _, w in pos]), beta)
    for t, w in neg:
        meaning -= -w * np.maximum(0, space.zsims(t))
    modifiers = sum((g * _col(lex["axes"][k]) for k, g in dials.items() if g), np.zeros(n))
    modifiers += softness * _col(lex["soft"])
    modifiers -= rarity * np.maximum(0, RARITY_PIVOT - np.array(lex["zipf"], dtype=np.float64)) / 2

    banned = {space.lemma[space.index[t]] for t, _ in pot if t in space.index}
    allowed = np.array([not lex["offensive"][i] and space.lemma[i] not in banned for i in range(n)])
    rows = np.flatnonzero(allowed)
    if pot:
        rows = rows[np.argsort(-meaning[rows], kind="stable")[:SHORTLIST]]
    final = meaning[rows] + modifiers[rows]
    out, seen = [], set()
    for j in np.argsort(-final, kind="stable"):
        r = rows[j]
        if space.lemma[r] in seen:
            continue
        seen.add(space.lemma[r])
        out.append((int(r), float(final[j])))
        if len(out) == limit:
            break
    return out
