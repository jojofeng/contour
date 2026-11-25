"""End-to-end build: lexicon -> sounds -> embeddings -> space -> dials -> files.

Usage (from the `pipeline/` folder):

    # Default: Qwen3-Embedding-0.6B (needs internet access to Hugging Face)
    python -m contour_pipeline.build --out ../web/public/data

    # Stand-in: static ConceptNet Numberbatch vectors (no model download needed)
    python -m contour_pipeline.build --backend static \
        --vectors-file conceptnet-numberbatch-17-06-300.gz \
        --model conceptnet-numberbatch-17.06 --out ../web/public/data

Output files (all read by the web app; see docs/data-format.md):
    meta.json       model, dimensions, quantisation scales, dial metadata
    lexicon.json    one column per field (text, lemma, frequency, sound, dial scores)
    vectors-*.bin   int8 vectors, row-major, split into shards of <= 16 MB
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .axes import build_axes
from .embedders import make_embedder
from .geometry import build_space, quantise
from .lexicon import build_lexicon
from .phonetics import analyse

SHARD_BYTES = 16 * 1024 * 1024


def offensive_words() -> set[str]:
    """better_profanity's word list. Entries are only *flagged*; the app hides them by default."""
    import better_profanity

    path = Path(better_profanity.__file__).parent / "profanity_wordlist.txt"
    return set(path.read_text().split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="sentence-transformers", choices=["sentence-transformers", "onnx", "static"])
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--onnx-folder")
    ap.add_argument("--vectors-file")
    ap.add_argument("--dims", type=int, default=256, help="dimensions kept after PCA")
    ap.add_argument("--drop-top", type=int, default=1, help="leading principal components removed (all-but-the-top)")
    ap.add_argument("--max-common-words", type=int, default=60_000)
    ap.add_argument("--max-phrases", type=int, default=20_000)
    ap.add_argument("--out", default="../web/public/data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("1/5 lexicon")
    lexicon = build_lexicon(args.max_common_words, args.max_phrases, offensive=offensive_words())
    print(f"    {len(lexicon):,} entries ({sum(e.is_phrase for e in lexicon):,} phrases)")

    print(f"2/5 embeddings with {args.model}")
    embedder = make_embedder(args.backend, args.model, args.onnx_folder, args.vectors_file)
    raw = embedder.embed([e.text for e in lexicon])
    has_vector = ~np.isnan(raw).any(axis=1)
    if not has_vector.all():
        lexicon = [e for e, ok in zip(lexicon, has_vector) if ok]
        raw = raw[has_vector]
        print(f"    kept {len(lexicon):,} entries that have vectors")

    print("3/5 pronunciations and sound features")
    sounds = [analyse(e.text) for e in lexicon]
    soft = np.array([s.softness if s else np.nan for s in sounds])
    soft_z = (soft - np.nanmean(soft)) / np.nanstd(soft)

    print("4/5 geometry and dials")
    space = build_space(raw, dims=args.dims, drop_top=args.drop_top)
    index = {e.text: i for i, e in enumerate(lexicon)}
    axes = build_axes(space.vectors, index)
    for a in axes:
        print(f"    {a.key:13s} leave-one-out accuracy {a.loo_accuracy:.0%} ({a.seeds_used} seeds)")

    print("5/5 writing files")
    q, scale = quantise(space.vectors)
    for old in out.glob("vectors-*.bin"):
        old.unlink()
    flat = q.tobytes()
    shards = []
    for i, start in enumerate(range(0, len(flat), SHARD_BYTES)):
        name = f"vectors-{i}.bin"
        (out / name).write_bytes(flat[start : start + SHARD_BYTES])
        shards.append(name)

    r2 = lambda x: None if x is None or np.isnan(x) else round(float(x), 2)
    columns = {
        "text": [e.text for e in lexicon],
        "lemma": [e.lemma if e.lemma != e.text else "" for e in lexicon],  # "" = same as text
        "zipf": [e.zipf for e in lexicon],
        "phrase": [int(e.is_phrase) for e in lexicon],
        "offensive": [int("offensive" in e.flags) for e in lexicon],
        "phones": [" ".join(s.phones) if s else "" for s in sounds],
        "syllables": [s.syllables if s else 0 for s in sounds],
        "stress": [s.stress if s else "" for s in sounds],
        "soft": [r2(v) for v in soft_z],
        "first": [s.first if s else "" for s in sounds],
        "rhyme": [s.rhyme if s else "" for s in sounds],
        "guessed": [int(bool(s) and s.source != "cmu") for s in sounds],
        "axes": {a.key: [r2(v) for v in a.z] for a in axes},
    }
    (out / "lexicon.json").write_text(json.dumps(columns, separators=(",", ":")))

    meta = {
        "model": args.model,
        "backend": args.backend,
        "stand_in": args.backend != "sentence-transformers",
        "count": len(lexicon),
        "dims": args.dims,
        "drop_top": args.drop_top,
        "explained_variance": round(space.explained, 3),
        "scale": [float(s) for s in scale],
        "shards": shards,
        "axes": [
            {"key": a.key, "label": a.label, "negative": a.negative_label, "positive": a.positive_label,
             "description": a.description, "loo_accuracy": a.loo_accuracy}
            for a in axes
        ],
        "built": time.strftime("%Y-%m-%d"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    total = sum(os.path.getsize(out / f) for f in [*shards, "lexicon.json", "meta.json"])
    print(f"done in {time.time() - t0:.0f}s; {total / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
