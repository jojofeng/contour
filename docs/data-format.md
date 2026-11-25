# Data format

`python -m contour_pipeline.build` writes three kinds of file to `web/public/data/`. The app loads them with `web/src/data.ts`.

## meta.json

| Field | Meaning |
|---|---|
| `model`, `backend` | which embedding model and backend built the space |
| `prompt` | instruction placed before every entry when embedding ("" if none) |
| `stand_in` | true for non-default backends; the app then shows a notice |
| `count`, `dims` | rows and dimensions of the vector matrix |
| `drop_top`, `explained_variance` | geometry clean-up settings and the share of variance kept |
| `scale` | one float per dimension: value = int8 × scale / 127 |
| `shards` | vector files to concatenate, in order |
| `axes` | the dials: key, labels, description, leave-one-out accuracy |

## vectors-N.bin

Raw int8, row-major, `count × dims` across all shards (each ≤ 16 MB, so static hosts and CDNs serve them happily). Row *r* is the entry `lexicon.text[r]`.

For hosts that refuse binary files, `npm run build && npm run pack:text` rewrites them as base64 text shards (`vectors-N.txt`); the app reads either form.

## lexicon.json

Columnar: every field is an array of length `count`.

| Field | Meaning |
|---|---|
| `text` | the entry, lowercase ("luminous", "run out") |
| `lemma` | inflection-group key; `""` means the same as `text` |
| `zipf` | log₁₀ frequency per billion words (wordfreq); phrases are estimated |
| `phrase`, `offensive` | 0/1 flags |
| `phones` | ARPAbet with stress digits; `""` if unknown |
| `syllables`, `stress` | syllable count; one stress digit per syllable |
| `soft` | z-scored softness, or null if unknown |
| `first`, `rhyme` | first phoneme; last stressed vowel to the end |
| `guessed` | 1 if the pronunciation came from the g2p model |
| `axes.<key>` | z-scored projection onto each dial's direction |
