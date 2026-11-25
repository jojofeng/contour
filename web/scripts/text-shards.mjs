// Convert dist/data/vectors-*.bin to base64 text shards (vectors-*.txt, each
// under ~12 MB) and update meta.json. Only needed for static hosts that refuse
// binary files; GitHub Pages serves the .bin files as they are.
//   npm run build && node scripts/text-shards.mjs
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";

const CHUNK = 9 * 1024 * 1024; // bytes of binary per text shard (base64 adds a third)
const dir = new URL("../dist/data/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("meta.json", dir), "utf8"));
const binary = Buffer.concat(meta.shards.map((name) => readFileSync(new URL(name, dir))));
meta.shards.forEach((name) => unlinkSync(new URL(name, dir)));
meta.shards = [];
for (let i = 0, off = 0; off < binary.length; i++, off += CHUNK) {
  const name = `vectors-${i}.txt`;
  writeFileSync(new URL(name, dir), binary.subarray(off, off + CHUNK).toString("base64"));
  meta.shards.push(name);
}
writeFileSync(new URL("meta.json", dir), JSON.stringify(meta, null, 1));
console.log("text shards:", meta.shards.join(", "));
