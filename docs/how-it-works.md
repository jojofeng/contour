# How Contour works

Everything here is implemented twice and kept identical: `web/src/engine.ts` (the app) and `pipeline/contour_pipeline/space.py` (experiments). `web/test/engine.test.ts` checks that they produce the same ranking.

## 1. The space

The pipeline embeds every vocabulary entry ("luminous", "run out", "at a loss") as bare text with one embedding model, then:

1. **Centres** the vectors (subtracts the vocabulary mean). Raw embedding spaces are anisotropic: all vectors lean one way, so every cosine similarity looks high. With the small sentence encoder we tried, almost every pair scored 0.75–0.85.
2. **Removes the top principal component and keeps the next 256** (Mu & Viswanath 2018, "All-but-the-top"). The leading direction mostly encodes frequency, not meaning.
3. **Re-normalises** to unit length and **quantises** to int8 with one scale per dimension. About 100k × 256 bytes ≈ 25 MB.

All of these steps are linear, so adding and averaging vectors still means what it meant in the original space.

## 2. Similarity in standard units

For a pot item *i* with unit vector *qᵢ*, every candidate *c* gets

> sᵢ(c) = v_c · qᵢ  zᵢ(c) = (sᵢ(c) − mean) / sd

where mean and sd are taken over the whole vocabulary. *zᵢ* reads as "how many standard deviations closer than a typical word". This puts every pot item on the same scale, which raw cosines are not: some words have many close neighbours and some have none.

## 3. Combining the pot: blend versus fit-every-one

For positive items with weights *wᵢ* (normalised to sum to 1):

> fit(c) = −(1/β) · ln Σᵢ wᵢ · exp(−β · zᵢ(c))

* **β = 0** gives the weighted mean Σ wᵢ zᵢ. This is equivalent to the classic "average the vectors" word arithmetic. It finds words *between* your words, and a word very close to one item can win while ignoring the others.
* **Large β** approaches minᵢ zᵢ. A word scores only as well as its *worst* match, so it must fit every item. This is a soft AND.

The slider "How to combine them" sets β from 0 to 4.

## 4. Steering away

An item with a negative weight subtracts |w| · max(0, zⱼ(c)). Being close to it costs points; being far from it earns nothing. "Difficult, but not tedious" removes the tedious sense without rewarding words that are merely unrelated to tedium.

## 5. Two-stage ranking

1. **Meaning:** score = fit − avoid. Keep the 500 best entries (`SHORTLIST`).
2. **Modifiers:** add the dials, sound and rarity terms, then re-rank those 500.

Without the shortlist, a strong dial would pull in words that have nothing to do with the pot. In testing, turning softness up surfaced "lo", "know" and "no" for a sky query. With an empty pot, the dials rank the whole vocabulary, which is useful for exploring ("the most technical words").

## 6. Dials: attributes as directions

Each dial is a direction in the space, estimated from seed words in `pipeline/data/axes.json` ("semantic projection", cf. Grand et al. 2022, *Nature Human Behaviour*):

* **Pairs** (warm → scorching, buy → purchase): the direction is the mean of the differences. Because each pair shares a topic, the topic cancels out and only the attribute remains.
* **Sets** (abstract words vs concrete words): the direction is the difference of the two means.

Every entry's projection onto the direction is z-scored. A dial at *g* ∈ [−1, 1] adds *g* × that score. Each dial's quality is reported as leave-one-out accuracy: the share of seed items placed on the correct side when that item was left out of the estimate.

## 7. Sound

Pronunciations come from the CMU Pronouncing Dictionary. Words it lacks get a pronunciation guessed by g2p_en's neural model, shown with "~" in the app. **Softness** is the stress-weighted mean of the per-phoneme weights in `pipeline/data/sound_weights.json`, then z-scored. The *direction* of each weight follows the sound-symbolism literature: m, n, l, b and rounded back vowels read as round; k, t, p and front vowels like /i/ read as spiky. The *magnitudes* are a judgement call. Syllables, starting sound and rhyme (last stressed vowel to the end) are exact filters.

## 8. Rarity

Rare words stay in play but lose r × max(0, 3.5 − zipf)/2, where zipf is log₁₀ frequency per billion words (wordfreq). At full strength a word that never appears in wordfreq loses 1.75σ.

## 9. Phrases not in the vocabulary

Text not in the vocabulary is split into its longest vocabulary units, matching multiword expressions first. The units are averaged with smooth-inverse-frequency weights a / (a + p(unit)), with a = 10⁻³ (Arora, Liang & Ma 2017), so "of" counts far less than "running". The app labels such items as approximated. Embedding them exactly needs the model running live (see the roadmap in the README).

## 10. The geometry view

* **One pot word:** suggestions are placed radially; closer means more similar.
* **Two pot words:** an exact scatter of z₁ against z₂. The dashed diagonal is "equally close to both".
* **Three or more:** an anchored layout (RadViz). Pot words sit on a ring, and each suggestion sits at the softmax(1.4 · zᵢ)-weighted average of the anchors. Unlike t-SNE or UMAP this layout is stable: moving a weight moves only the points whose numbers changed. Like any 2-D view of 256 dimensions it is a summary; the table under the map has the exact numbers.

## What the score is not

The score is not a probability or a confidence. A score of 5σ means "much closer than typical", which says nothing about whether this is *your* word. The "This one" log is the start of measuring that.
