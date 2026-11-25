"""Build the candidate vocabulary ("lexicon").

The app can only suggest what is in this list, so the choices here matter as
much as the embedding model. The lexicon is the union of:

1. Common single words, from `wordfreq`'s frequency-ranked English list,
   kept only if WordNet recognises them (directly or as an inflection) or the
   CMU Pronouncing Dictionary lists them. This drops most junk tokens.
2. Rare single words: every lowercase WordNet lemma. Rare words stay "in
   play"; the app's rarity dial merely down-weights them.
3. Multiword expressions: lowercase WordNet lemmas of 2-4 words ("run out",
   "at a loss", "bitter end"). Capitalised lemmas are proper nouns or
   taxonomic names and are skipped.

Each entry records a `lemma` key so the app can group inflections
("difficulty" / "difficulties") and exclude forms of words already in the pot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

WORD_RE = re.compile(r"^[a-z]+(?:[-'][a-z]+)*$")
_POS_ORDER = ("n", "v", "a", "r")  # morphy tries parts of speech in this order


@dataclass
class Entry:
    text: str            # display form, e.g. "run out"
    lemma: str           # grouping key for inflections, e.g. "run out"
    zipf: float          # log10 frequency per billion words (wordfreq); 0 if unknown
    is_phrase: bool
    flags: list[str] = field(default_factory=list)  # e.g. ["offensive"]


def ensure_wordnet() -> None:
    """Download NLTK's WordNet data on first use (about 10 MB)."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def lemma_of(word: str) -> str | None:
    """Return the WordNet base form of `word`, or None if WordNet doesn't know it."""
    for pos in _POS_ORDER:
        base = wn.morphy(word, pos)
        if base is not None:
            return base.replace("_", " ")
    return None


def _phrase_zipf(words: list[str]) -> float:
    """Rough frequency for a phrase.

    wordfreq's own multiword estimate is unreliable for collocations, so we use
    the rarest word's frequency minus a fixed penalty (phrases are rarer than
    their parts). Only used to rank phrases and to drive the rarity dial.
    """
    return max(0.0, min(zipf_frequency(w, "en") for w in words) - 1.5)


def build_lexicon(
    max_common_words: int = 60_000,
    max_phrases: int = 20_000,
    min_phrase_word_zipf: float = 2.5,
    offensive: set[str] | None = None,
) -> list[Entry]:
    offensive = offensive or set()
    ensure_wordnet()
    from .phonetics import cmu_dict  # local import: heavy

    cmu = cmu_dict()
    entries: dict[str, Entry] = {}

    def add(text: str, lemma: str, zipf: float, is_phrase: bool) -> None:
        if text in entries:
            return
        flags = ["offensive"] if any(w in offensive for w in text.split()) else []
        entries[text] = Entry(text, lemma, round(zipf, 2), is_phrase, flags)

    # 1. common words (frequency ranked)
    for w in top_n_list("en", 200_000):
        if len(entries) >= max_common_words:
            break
        if not WORD_RE.match(w) or w.endswith("'s") or (len(w) < 2 and w not in ("a", "i")):
            continue
        lemma = lemma_of(w)
        # Words WordNet doesn't know are kept only if common and pronounceable
        # (function words, contractions); this filters most names and junk.
        if lemma is None and (w not in cmu or zipf_frequency(w, "en") < 4.0):
            continue
        add(w, lemma or w, zipf_frequency(w, "en"), False)

    # 2. rare WordNet single words, 3. WordNet multiword expressions
    phrases: dict[str, float] = {}
    for synset in wn.all_synsets():
        for lem in synset.lemmas():
            name = lem.name()
            if name != name.lower():
                continue  # proper noun or taxonomic name
            text = name.replace("_", " ")
            words = text.split()
            if len(words) == 1:
                if WORD_RE.match(text) and len(text) > 2:
                    add(text, lemma_of(text) or text, zipf_frequency(text, "en"), False)
            elif 2 <= len(words) <= 4 and all(WORD_RE.match(x) for x in words):
                if min(zipf_frequency(x, "en") for x in words) >= min_phrase_word_zipf:
                    # SemCor usage counts give a small boost to phrases people actually use
                    phrases[text] = max(phrases.get(text, 0.0), _phrase_zipf(words) + 0.25 * min(lem.count(), 4))

    for text, score in sorted(phrases.items(), key=lambda kv: -kv[1])[:max_phrases]:
        add(text, text, _phrase_zipf(text.split()), True)

    return list(entries.values())
