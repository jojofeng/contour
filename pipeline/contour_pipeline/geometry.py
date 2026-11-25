"""Clean up the raw embedding geometry and pack it for the browser.

Raw embedding spaces are anisotropic: every vector leans toward a shared
common direction, so all cosine similarities bunch together (0.75-0.85 for
the stand-in model). Following Mu & Viswanath (2018, "All-but-the-top"), we

1. subtract the lexicon's mean vector,
2. optionally remove the top `drop_top` principal components (which mostly
   encode frequency and other non-semantic signals), then
3. project onto the next `dims` principal components and re-normalise.

Steps 2-3 are one PCA. Every step is linear, so composition stays meaningful.
Vectors are then quantised to int8 with one scale per dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Space:
    vectors: np.ndarray        # (n, dims) float32, unit length
    mean: np.ndarray           # (orig_dims,) subtracted before projection
    components: np.ndarray     # (dims, orig_dims) projection basis
    explained: float           # share of centred variance kept


def build_space(raw: np.ndarray, dims: int = 256, drop_top: int = 0) -> Space:
    mean = raw.mean(axis=0)
    centred = raw - mean
    # PCA via SVD on a sample is plenty accurate and keeps memory modest
    rng = np.random.default_rng(0)
    sample = centred[rng.choice(len(centred), size=min(len(centred), 40_000), replace=False)]
    _, s, vt = np.linalg.svd(sample, full_matrices=False)
    keep = slice(drop_top, drop_top + dims)
    components = vt[keep]
    variance = s**2
    explained = float(variance[keep].sum() / variance.sum())
    projected = centred @ components.T
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    return Space(projected.astype(np.float32), mean.astype(np.float32), components.astype(np.float32), explained)


def quantise(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 per-dimension quantisation. Reconstruct with q * scale / 127."""
    scale = np.abs(vectors).max(axis=0) + 1e-12
    q = np.clip(np.round(vectors / scale * 127), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)
