"""Write web/test/parity.json: top results computed in Python for the TS engine test to match.

    cd pipeline && python -m tests.make_parity_fixture
"""
import json
from pathlib import Path

import numpy as np

from contour_pipeline.space import LoadedSpace, rank_rows

DATA = Path(__file__).resolve().parents[2] / "web" / "public" / "data"
CASES = [
    {"pot": [["difficult", 1.0], ["exciting", 0.8]], "beta": 2.0, "dials": {"technicality": 0.6, "valence": 0.5}, "softness": 0, "rarity": 0.5},
    {"pot": [["laugh", 1.0], ["quiet", 1.0]], "beta": 0.0, "dials": {}, "softness": 0.0, "rarity": 0.0},
    {"pot": [["sky", 1.0], ["clear", 1.0], ["glow", 0.7], ["cloud", -1.0]], "beta": 4.0, "dials": {"intensity": -0.4}, "softness": 0.8, "rarity": 0.3},
]

space = LoadedSpace.load(DATA)
out = []
for case in CASES:
    rows = rank_rows(space, [tuple(p) for p in case["pot"]], case["beta"], case["dials"], case["softness"], case["rarity"])
    top = [{"text": space.lex["text"][r], "score": round(score, 4)} for r, score in rows]
    out.append({**case, "top": top})
    print(" + ".join(p[0] for p in case["pot"]), "->", [t["text"] for t in top])
Path(__file__).resolve().parents[2].joinpath("web", "test", "parity.json").write_text(json.dumps(out, indent=1))
