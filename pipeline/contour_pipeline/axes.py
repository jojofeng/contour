"""Attribute dials as directions in the space ("semantic projection").

A dial such as valence is the direction from one pole to the other,
estimated from hand-picked seed words in `data/axes.json` (cf. Grand et al.,
2022, Nature Human Behaviour). Two seed formats are supported:

* `pairs`: [[mild, intense], ...]. The direction is the mean difference
  between paired words, which cancels out the topic and isolates the attribute
  (e.g. warm -> scorching, like -> adore).
* `negative` / `positive`: two word sets. The direction is the difference of
  their means.

Every lexicon entry's projection is standardised (z-score over the lexicon) so
dials are on a common scale. Each dial gets a quality score: a leave-one-out
accuracy, the share of seed items placed on the correct side when that item
was left out of the estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Axis:
    key: str
    label: str
    negative_label: str
    positive_label: str
    description: str
    z: np.ndarray            # standardised projection of every lexicon entry
    loo_accuracy: float
    seeds_used: int


def load_axis_specs() -> list[dict]:
    return json.loads((DATA_DIR / "axes.json").read_text())["axes"]


def build_axes(vectors: np.ndarray, index: dict[str, int]) -> list[Axis]:
    axes = []
    for spec in load_axis_specs():
        if "pairs" in spec:
            pairs = [(index[a], index[b]) for a, b in spec["pairs"] if a in index and b in index]
            diffs = np.stack([vectors[b] - vectors[a] for a, b in pairs])
            direction = diffs.mean(axis=0)
            # leave-one-out: is the held-out pair ordered correctly?
            correct = 0
            for i, (a, b) in enumerate(pairs):
                d = (diffs.sum(axis=0) - diffs[i]) / (len(pairs) - 1)
                correct += float(vectors[b] @ d > vectors[a] @ d)
            loo, used = correct / len(pairs), len(pairs)
        else:
            neg = [index[w] for w in spec["negative"] if w in index]
            pos = [index[w] for w in spec["positive"] if w in index]
            direction = vectors[pos].mean(axis=0) - vectors[neg].mean(axis=0)
            correct = 0
            for side, members in ((1, pos), (-1, neg)):
                for m in members:
                    p = [i for i in pos if i != m]
                    n = [i for i in neg if i != m]
                    d = vectors[p].mean(axis=0) - vectors[n].mean(axis=0)
                    midpoint = (vectors[p].mean(axis=0) + vectors[n].mean(axis=0)) / 2
                    correct += float(side * ((vectors[m] - midpoint) @ d) > 0)
            loo, used = correct / (len(pos) + len(neg)), len(pos) + len(neg)

        direction /= np.linalg.norm(direction)
        proj = vectors @ direction
        z = (proj - proj.mean()) / proj.std()
        axes.append(
            Axis(spec["key"], spec["label"], spec["negative_label"], spec["positive_label"],
                 spec.get("description", ""), z.astype(np.float32), round(loo, 3), used)
        )
    return axes
