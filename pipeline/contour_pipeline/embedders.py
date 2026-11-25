"""Embedding backends.

Every backend turns a list of strings into L2-normalised float32 vectors.
The rest of the pipeline never needs to know which model produced them.

* `sentence-transformers` (default): any Hugging Face embedding model. The
  default is Qwen/Qwen3-Embedding-0.6B (Apache-2.0), the strongest open model
  that embeds the whole lexicon on an ordinary CPU in well under an hour.
  The 4B and 8B variants work the same way if you have the compute.
* `onnx`: a local folder holding `model.onnx` + `tokenizer.json`.
* `static`: a table of pre-trained word vectors (word2vec text format, .gz or
  .txt, or an .npz with `keys` and `vectors`). Used for the stand-in build
  with ConceptNet Numberbatch. Entries missing from the table come back as NaN
  rows and are dropped from the lexicon by build.py.

Pot items and candidates are the same kind of thing, so every entry is
embedded the same way and the space is used symmetrically. Instruction-aware
models (Qwen3-Embedding) get a short instruction before each entry, because a
bare word out of context tends to land near words that merely look or sound
alike: in our first Qwen3 build, "false + name" gave "no, none, null".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Qwen3-Embedding's instruction format is "Instruct: {task}\nQuery:{text}".
QWEN3_WORD_PROMPT = "Instruct: Represent the meaning of this English word or phrase\nQuery:"


def default_prompt(model_id: str) -> str:
    return QWEN3_WORD_PROMPT if "qwen3-embedding" in model_id.lower() else ""


class Embedder(ABC):
    model_id: str

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B", prompt: str = ""):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.prompt = prompt
        # float32: on most CPUs bfloat16 is emulated, which is slower and slightly less precise
        self.model = SentenceTransformer(model_id, model_kwargs={"dtype": torch.float32})

    def embed(self, texts, batch_size=256):
        out = self.model.encode(texts, prompt=self.prompt or None, batch_size=batch_size,
                                show_progress_bar=True, convert_to_numpy=True)
        return _normalise(out)


class OnnxEmbedder(Embedder):
    def __init__(self, folder: str, model_id: str, output: str = "sentence_embedding"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder_path = Path(folder)
        self.model_id = model_id
        self.output = output
        self.tokenizer = Tokenizer.from_file(str(folder_path / "tokenizer.json"))
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(folder_path / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def embed(self, texts, batch_size=256):
        chunks = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[start : start + batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            chunks.append(self.session.run([self.output], {k: v for k, v in feeds.items() if k in self.inputs})[0])
            if (start // batch_size) % 40 == 0:
                print(f"  embedded {start + len(enc):,} / {len(texts):,}", flush=True)
        return _normalise(np.concatenate(chunks))


class StaticVectorsEmbedder(Embedder):
    def __init__(self, path: str, model_id: str, key_prefix: str = "/c/en/"):
        self.model_id = model_id
        if path.endswith(".npz"):
            data = np.load(path)
            keys, vectors = list(data["keys"]), data["vectors"]
        else:
            keys, vectors = _read_word2vec_text(path, key_prefix)
        self.index = {k: i for i, k in enumerate(keys)}
        self.vectors = vectors.astype(np.float32)

    def embed(self, texts, batch_size=256):
        out = np.full((len(texts), self.vectors.shape[1]), np.nan, dtype=np.float32)
        for i, t in enumerate(texts):
            j = self.index.get(t)
            if j is not None:
                out[i] = self.vectors[j]
        found = ~np.isnan(out[:, 0])
        out[found] = _normalise(out[found])
        print(f"  {found.sum():,} / {len(texts):,} entries have vectors")
        return out


def _read_word2vec_text(path: str, key_prefix: str) -> tuple[list[str], np.ndarray]:
    """word2vec text format; keys like /c/en/run_out become "run out"."""
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    keys, rows = [], []
    with opener(path, "rt", encoding="utf-8") as f:
        f.readline()  # header: count dims
        for line in f:
            if key_prefix and not line.startswith(key_prefix):
                continue
            parts = line.rstrip().split(" ")
            keys.append(parts[0][len(key_prefix):].replace("_", " "))
            rows.append(np.array(parts[1:], dtype=np.float32))
    return keys, np.stack(rows)


def make_embedder(backend: str, model: str, onnx_folder: str | None = None, vectors_file: str | None = None,
                  prompt: str = "") -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model, prompt)
    if backend == "onnx":
        if not onnx_folder:
            raise ValueError("--onnx-folder is required for the onnx backend")
        return OnnxEmbedder(onnx_folder, model)
    if backend == "static":
        if not vectors_file:
            raise ValueError("--vectors-file is required for the static backend")
        return StaticVectorsEmbedder(vectors_file, model)
    raise ValueError(f"unknown backend {backend!r}")
