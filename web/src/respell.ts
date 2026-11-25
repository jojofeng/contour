/**
 * Turn ARPAbet ("L UW1 M AH0 N AH0 S") into a readable respelling
 * ("LOO·muh·nuhs"): stressed syllable in capitals, syllables split before the
 * consonant that precedes each vowel.
 */

const SOUND: Record<string, string> = {
  AA: "ah", AE: "a", AH: "uh", AO: "aw", AW: "ow", AY: "eye", EH: "eh", ER: "ur", EY: "ay",
  IH: "ih", IY: "ee", OW: "oh", OY: "oy", UH: "uu", UW: "oo",
  B: "b", CH: "ch", D: "d", DH: "th", F: "f", G: "g", HH: "h", JH: "j", K: "k", L: "l", M: "m",
  N: "n", NG: "ng", P: "p", R: "r", S: "s", SH: "sh", T: "t", TH: "th", V: "v", W: "w", Y: "y",
  Z: "z", ZH: "zh",
};

export function respell(phones: string): string {
  if (!phones) return "";
  const ps = phones.split(" ");
  const isVowel = (p: string) => /\d$/.test(p);
  const vowelAt = ps.map(isVowel);
  // syllable boundaries: before the consonant directly preceding each non-initial vowel
  const starts: number[] = [];
  let seenVowel = false;
  ps.forEach((_, i) => {
    if (!vowelAt[i]) return;
    if (!seenVowel) { starts.push(0); seenVowel = true; return; }
    starts.push(i > 0 && !vowelAt[i - 1] ? i - 1 : i);
  });
  if (!starts.length) return ps.map((p) => SOUND[p] ?? p.toLowerCase()).join("");
  return starts
    .map((s, k) => {
      const chunk = ps.slice(s, starts[k + 1] ?? ps.length);
      const stressed = chunk.some((p) => p.endsWith("1"));
      const text = chunk.map((p) => SOUND[p.replace(/\d$/, "")] ?? p.toLowerCase()).join("");
      return stressed && starts.length > 1 ? text.toUpperCase() : text;
    })
    .join("·");
}
