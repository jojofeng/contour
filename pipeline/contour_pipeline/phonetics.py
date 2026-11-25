"""Pronunciations and transparent sound features.

Embeddings barely encode how words sound, so sound is handled by a separate,
fully inspectable channel:

* Pronunciations come from the CMU Pronouncing Dictionary. Words it lacks get a
  pronunciation guessed by g2p_en's small neural model (flagged `source="g2p"`).
* `softness` is a bouba/kiki-style score: the stress-weighted mean of
  per-phoneme weights in `data/sound_weights.json`. The weights follow the
  sound-symbolism literature (sonorants, voiced stops and rounded back vowels
  read as "round"; voiceless stops and front unrounded vowels as "spiky"),
  but the exact numbers are a judgement call. Edit the file to change them.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_VOWEL = re.compile(r"^[A-Z]{2}[012]$")


@dataclass
class Sound:
    phones: list[str]     # ARPAbet with stress digits, e.g. ["L", "UW1", "M", "AH0", "N", "AH0", "S"]
    syllables: int
    stress: str           # one digit per syllable, e.g. "100"
    softness: float       # raw score in roughly [-1, 1]; the app standardises it
    first: str            # first phoneme without stress, e.g. "L"
    rhyme: str            # last stressed vowel to the end, e.g. "UW1 M AH0 N AH0 S"
    source: str           # "cmu" or "g2p"


@lru_cache(maxsize=1)
def cmu_dict() -> dict[str, list[list[str]]]:
    import cmudict

    return cmudict.dict()


@lru_cache(maxsize=1)
def sound_weights() -> dict[str, float]:
    weights = json.loads((DATA_DIR / "sound_weights.json").read_text())
    return {k: float(v) for k, v in weights["phonemes"].items()}


@lru_cache(maxsize=1)
def _g2p():
    """g2p_en's neural model, loaded without the NLTK downloads it tries at import."""
    import nltk

    real_download = nltk.download
    nltk.download = lambda *args, **kwargs: True  # g2p_en tries to fetch data we don't need
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import g2p_en.g2p as g2p_module
    finally:
        nltk.download = real_download
    g2p_module.cmudict = types.SimpleNamespace(dict=lambda: {})  # we do the CMU lookup ourselves
    return g2p_module.G2p()


def pronounce_word(word: str) -> tuple[list[str], str] | None:
    cmu = cmu_dict()
    key = word.replace("-", "")
    if word in cmu:
        return cmu[word][0], "cmu"
    if key in cmu:
        return cmu[key][0], "cmu"
    letters = re.sub(r"[^a-z]", "", word)
    if not letters:
        return None
    phones = [p for p in _g2p().predict(letters) if p not in ("<pad>", "<unk>", "<s>", "</s>")]
    return (phones, "g2p") if phones else None


def analyse(text: str) -> Sound | None:
    phones: list[str] = []
    source = "cmu"
    for word in text.split():
        result = pronounce_word(word)
        if result is None:
            return None
        word_phones, word_source = result
        phones += word_phones
        if word_source != "cmu":
            source = word_source

    weights = sound_weights()
    total = weight_sum = 0.0
    stress = ""
    for p in phones:
        base = p.rstrip("012")
        w = 1.0
        if _VOWEL.match(p):
            stress += p[-1]
            w = 1.5 if p.endswith("1") else 1.0  # stressed vowels dominate the impression
        total += w * weights.get(base, 0.0)
        weight_sum += w

    last_stressed = max((i for i, p in enumerate(phones) if p.endswith("1")), default=None)
    if last_stressed is None:
        last_stressed = max((i for i, p in enumerate(phones) if _VOWEL.match(p)), default=0)

    return Sound(
        phones=phones,
        syllables=len(stress),
        stress=stress,
        softness=round(total / weight_sum, 3) if weight_sum else 0.0,
        first=phones[0].rstrip("012") if phones else "",
        rhyme=" ".join(p.rstrip("012") for p in phones[last_stressed:]),
        source=source,
    )
