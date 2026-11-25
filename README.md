# Contour

Find a word by the shape of what you mean.

Add words or phrases that point toward the meaning you're after and weight them. Steer away from senses you don't want, nudge attributes (more positive, more intense, more technical, more formal, more concrete), and (optionally) shape how it sounds: softer or sharper, a set number of syllables, a starting sound, a rhyme. The sound controls are switched off for now; set `SOUND_CONTROLS = true` in `web/src/main.ts` to bring them back. Contour ranks about 100,000 English words and phrases by how well they fit, and shows its working: every score breaks down into its parts, and "Show the geometry" draws where the suggestions sit relative to your words.

No language model writes the suggestions. They come from distances in an embedding space plus a few transparent, editable sound and frequency features. The point is to test whether meaning has usable geometric structure.

## Repository layout

```
pipeline/                 Python: builds the vocabulary space the app loads
  contour_pipeline/
    lexicon.py            which words and phrases are candidates
    phonetics.py          pronunciations and the bouba/kiki "softness" score
    embedders.py          embedding backends (Qwen3 via sentence-transformers, ONNX, static vectors)
    geometry.py           centring, PCA, int8 quantisation
    axes.py               attribute dials as directions in the space
    build.py              runs everything and writes web/public/data/
    space.py              loads the output and scores exactly like the app (for experiments)
    evaluate.py           recall on tip-of-the-tongue test cases
  data/
    axes.json             seed words for each dial: edit to add or change dials
    sound_weights.json    per-phoneme softness weights: edit to taste
    eval_cases.json       test cases: add your own real tip-of-the-tongue moments
  tests/make_parity_fixture.py
web/                      TypeScript + Vite static site
  src/engine.ts           the scoring maths (pure functions)
  src/data.ts             loads the pipeline's files
  src/hood.ts             the geometry view: map + itemised score
  src/main.ts             UI wiring, persistence, picks log
  test/engine.test.ts     checks the TS engine ranks exactly like the Python reference
docs/how-it-works.md      the formulas and the reasoning behind them
docs/data-format.md       what the pipeline writes
.github/workflows/        free build + deploy to GitHub Pages
```

## Put it online (free)

1. Create a GitHub repository and push this folder to its `main` branch.
2. In the repository: **Settings → Pages → Source: GitHub Actions**.
3. The workflow in `.github/workflows/build-and-deploy.yml` runs on every push. It embeds the vocabulary with **Qwen/Qwen3-Embedding-0.6B** on GitHub's free runner. The first run takes roughly 1–2 hours of CPU time; later runs reuse the cached result unless `pipeline/` changes. It then logs the evaluation, runs the tests and publishes the site. A push during a build waits for it to finish rather than cancelling it.

To try a different model, open **Actions → Build and deploy → Run workflow** and enter any Hugging Face embedding model ID (for example `Qwen/Qwen3-Embedding-4B` if you can wait longer).

## Run it locally

```bash
# 1. Build a space (pick one)
cd pipeline
pip install -r requirements.txt   # plus torch for the Qwen3 default, see the file
python -m contour_pipeline.build --out ../web/public/data                        # Qwen3-Embedding-0.6B

# or the offline stand-in used for the preview: ConceptNet Numberbatch static vectors
#   download conceptnet-numberbatch-17-06-300.gz from
#   https://github.com/RaRe-Technologies/gensim-data/releases/tag/conceptnet-numberbatch-17-06-300
python -m contour_pipeline.build --backend static --vectors-file conceptnet-numberbatch-17-06-300.gz \
    --model "ConceptNet Numberbatch 17.06" --out ../web/public/data

python -m contour_pipeline.evaluate --show       # how well does this space find words?
python -m tests.make_parity_fixture              # refresh the cross-language test

# 2. Run the app
cd ../web
npm install
npm run dev          # http://localhost:5173
npm test             # engine parity tests
npm run build        # static site in web/dist
```

## What we've learned so far

Measured on the 42 cases in `pipeline/data/eval_cases.json`: a few words go into the pot with equal weight, and a case is a hit if one of its target words appears in the top *k*.

| Space | Combine | hit@1 | hit@10 | hit@50 | MRR |
|---|---|---|---|---|---|
| IBM granite-embedding-small-english-r2 (sentence encoder) | soft-AND β=2 | 7% | 24% | 31% | 0.13 |
| ConceptNet Numberbatch 17.06 (static word vectors) | blend β=0 | 12% | 62% | 93% | 0.27 |
| ConceptNet Numberbatch 17.06 | soft-AND β=2 | 21% | 62% | 88% | 0.33 |
| Qwen3-Embedding-0.6B | — | run the workflow; the log prints this table | | | |

* The small sentence encoder is poor on single words. Its neighbours share *spelling* rather than meaning ("fear + heights → frights, footlights"). This is a known weakness of models trained on sentences, and the reason this table exists: whether Qwen3's space beats static vectors on single words is an open, testable question.
* Requiring a word to fit *every* pot word (soft-AND) improves the top ranks over plain averaging (MRR 0.27 → 0.33). That is a small but real point against a pure "concepts add like vectors" view.
* Every dial puts at least 96% of its held-out seed words on the right side in Numberbatch (82–92% in the small encoder).

## Roadmap

Planned next, in rough order:

1. **Exact phrase embedding in the browser.** Today a phrase that isn't in the vocabulary is approximated by a frequency-weighted average of its parts, and the UI says so. Running Qwen3 in the browser (transformers.js) would embed it exactly, at the cost of a roughly 600 MB one-time download. A small free serverless endpoint is the alternative.
2. **Context.** Re-rank the shortlist by how well each candidate fits a sentence you paste in.
3. **Real examples with citations.** Retrieved verbatim from open corpora (Wiktionary quotations, Project Gutenberg, Wikipedia, Semantic Scholar's open-access snippets), never generated.
4. **Compare spaces side by side.** The pipeline already supports several; the app could load two.
5. **Learn from your picks.** The "This one" log already records the pot, the settings and the rank of the chosen word.

## Licences of the data

Code: choose a licence before publishing. Data sources: wordfreq (Apache-2.0 code; data CC-BY-SA 4.0), WordNet (WordNet licence), CMU Pronouncing Dictionary (BSD-style), g2p_en (Apache-2.0), Qwen3-Embedding (Apache-2.0), ConceptNet Numberbatch (CC-BY-SA 4.0), better_profanity word list (MIT).
