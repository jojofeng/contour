"""Does the geometry actually find the word? Recall on tip-of-the-tongue cases.

    python -m contour_pipeline.evaluate --data ../web/public/data

For each case in data/eval_cases.json, the pot words get equal weight and the
candidates are ranked by the app's fit score (no dials, no rarity penalty),
excluding forms of the pot words. Reports, for several settings of the
blend <-> must-fit-all knob (beta):

    hit@1 / hit@10 / hit@50   share of cases with a target in the top k
    MRR                       mean reciprocal rank of the best-ranked target

A "raw centroid" baseline ranks by cosine to the mean of the pot vectors,
the textbook word-vector arithmetic, for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .space import LoadedSpace, fit_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def rank_of_targets(space: LoadedSpace, scores: np.ndarray, pot: list[str], targets: list[str]) -> int | None:
    excluded = {space.lemma[space.index[p]] for p in pot if p in space.index} | set(pot)
    order = np.argsort(-scores)
    seen_lemmas: set[str] = set()
    rank = 0
    target_set = set(targets)
    for i in order[:5000]:
        lemma = space.lemma[i]
        if lemma in excluded or space.lex["text"][i] in excluded or space.lex["offensive"][i]:
            continue
        if lemma in seen_lemmas:  # the app shows one row per lemma group
            continue
        seen_lemmas.add(lemma)
        rank += 1
        if space.lex["text"][i] in target_set or lemma in target_set:
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../web/public/data")
    ap.add_argument("--show", action="store_true", help="print the top 8 for every case")
    args = ap.parse_args()

    space = LoadedSpace.load(args.data)
    cases = json.loads((DATA_DIR / "eval_cases.json").read_text())["cases"]
    usable = [c for c in cases if all(p in space.index for p in c["pot"])]
    print(f"model: {space.meta['model']}  |  {len(usable)}/{len(cases)} cases usable\n")

    settings = {"raw centroid": None, "blend (β=0)": 0.0, "β=1": 1.0, "β=2": 2.0, "must fit all (β=4)": 4.0}
    print(f"{'setting':22s} hit@1  hit@10  hit@50   MRR")
    for name, beta in settings.items():
        ranks = []
        for case in usable:
            if beta is None:
                c = np.mean([space.query_vector(p) for p in case["pot"]], axis=0)
                scores = space.vectors @ (c / np.linalg.norm(c))
            else:
                z = np.stack([space.zsims(p) for p in case["pot"]])
                scores = fit_score(z, np.ones(len(case["pot"])), beta)
            ranks.append(rank_of_targets(space, scores, case["pot"], case["targets"]))
        hit = lambda k: np.mean([r is not None and r <= k for r in ranks])
        mrr = np.mean([1 / r if r else 0 for r in ranks])
        print(f"{name:22s} {hit(1):5.0%} {hit(10):7.0%} {hit(50):7.0%}  {mrr:.3f}")

    if args.show:
        print()
        for case in usable:
            z = np.stack([space.zsims(p) for p in case["pot"]])
            scores = fit_score(z, np.ones(len(case["pot"])), 2.0)
            excluded = {space.lemma[space.index[p]] for p in case["pot"]}
            top, seen = [], set()
            for i in np.argsort(-scores):
                lem = space.lemma[i]
                if lem in excluded or lem in seen or space.lex["offensive"][i]:
                    continue
                seen.add(lem)
                top.append(space.lex["text"][i])
                if len(top) == 8:
                    break
            print(f"{' + '.join(case['pot']):28s} -> {', '.join(top)}")


if __name__ == "__main__":
    main()
