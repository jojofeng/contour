/**
 * The TypeScript engine must rank exactly like the Python reference
 * (pipeline/contour_pipeline/space.py). Regenerate the fixture after a data
 * rebuild with:  cd pipeline && python -m tests.make_parity_fixture
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { assembleSpace, type Meta, type RawLexicon } from "../src/data";
import { rank, softAnd, vectorFor, zSimilarities, type PotItem, type Settings } from "../src/engine";

const dataDir = resolve(__dirname, "../public/data");
const meta: Meta = JSON.parse(readFileSync(`${dataDir}/meta.json`, "utf8"));
const raw: RawLexicon = JSON.parse(readFileSync(`${dataDir}/lexicon.json`, "utf8"));
const shards = meta.shards.map((s) => {
  const b = readFileSync(`${dataDir}/${s}`);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
});
const space = assembleSpace(meta, raw, shards);
const cases = JSON.parse(readFileSync(resolve(__dirname, "parity.json"), "utf8"));

describe("softAnd", () => {
  it("is the weighted mean at beta = 0", () => {
    expect(softAnd([1, 3], [1, 1], 0)).toBeCloseTo(2);
  });
  it("approaches the minimum as beta grows", () => {
    expect(softAnd([1, 3], [1, 1], 50)).toBeCloseTo(1 + Math.log(2) / 50, 3);
  });
});

describe("parity with the Python reference", () => {
  for (const c of cases) {
    it(c.pot.map((p: [string, number]) => p[0]).join(" + "), () => {
      const pot: PotItem[] = c.pot.map(([text, weight]: [string, number], i: number) => {
        const v = vectorFor(space, text)!;
        return { id: String(i), text, weight, vector: v.vector, approx: v.approx, parts: v.parts };
      });
      const z = pot.map((p) => zSimilarities(space, p.vector));
      const settings: Settings = {
        beta: c.beta, dials: c.dials, softness: c.softness, syllables: 0, startsLike: "",
        rhymesWith: "", rarity: c.rarity, phrases: true, hideOffensive: true,
      };
      const got = rank(space, pot, z, settings, 10);
      const want: { text: string; score: number }[] = c.top;
      // Scores must agree position by position. Words must too, except where
      // Python (float64) and the browser (float32) could order a near-tie differently.
      got.forEach((r, i) => {
        expect(r.score).toBeCloseTo(want[i].score, 2);
        const text = space.lex.text[r.index];
        if (text !== want[i].text) {
          const j = want.findIndex((w) => w.text === text);
          const tieScore = j >= 0 ? want[j].score : want[want.length - 1].score;
          expect(Math.abs(tieScore - r.score)).toBeLessThan(0.01);
        }
      });
    });
  }
});
