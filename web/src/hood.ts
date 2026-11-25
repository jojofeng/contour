/**
 * "Show the geometry": a map of the suggestions relative to the pot, and the
 * itemised score of the selected suggestion.
 *
 * The map is an anchored layout (RadViz): pot words sit evenly on a ring and
 * each suggestion is placed at the average of the anchor positions, weighted
 * by softmax(z_i · τ) of its standardised similarity to each pot word. So a
 * point near an anchor is far more similar to that word than to the others;
 * one in the middle is equally similar to all. With a single pot word the map
 * becomes radial (distance from the word = 1 / similarity); with exactly two
 * it is an exact scatter plot of similarity to each word.
 *
 * Unlike t-SNE or UMAP this layout is stable: changing a weight never shuffles
 * the picture, it only moves the points whose numbers changed.
 */

import type { Space } from "./data";
import { SHORTLIST, type PotItem, type Result, type Settings } from "./engine";

const TAU = 1.4;      // softmax sharpness for the anchored layout
const LABELLED = 16;  // label the top N; the rest are dots

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const fmt = (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);

export function drawMap(svg: SVGSVGElement, space: Space, pot: PotItem[], results: Result[], selected: number, onSelect: (r: number) => void): void {
  const W = svg.clientWidth || 400, H = svg.clientHeight || 400;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 56;
  if (!pot.length || !results.length) { svg.innerHTML = ""; return; }

  let place: (res: Result, rank: number) => [number, number];
  let backdrop: string;

  if (pot.length === 2) {
    // Exact view for two words: x = similarity to the first, y = similarity to the second.
    const zs = results.flatMap((r) => r.z);
    const lo = Math.min(0, ...zs), hi = Math.max(...zs) + 0.3;
    const pad = 48, size = Math.min(W, H) - 2 * pad;
    const x0 = (W - size) / 2, y0 = (H + size) / 2;
    const sx = (z: number) => x0 + ((z - lo) / (hi - lo)) * size;
    const sy = (z: number) => y0 - ((z - lo) / (hi - lo)) * size;
    place = (res) => [sx(res.z[0]), sy(res.z[1])];
    const [a, b] = pot;
    backdrop =
      `<rect class="ring" x="${x0}" y="${y0 - size}" width="${size}" height="${size}"/>` +
      `<line class="spoke" x1="${sx(lo)}" y1="${sy(lo)}" x2="${sx(hi)}" y2="${sy(hi)}"/>` +
      `<g class="anchor${a.weight < 0 ? " away" : ""}"><text x="${x0 + size}" y="${y0 + 24}" text-anchor="end">closer to ${esc(a.text)} →</text></g>` +
      `<g class="anchor${b.weight < 0 ? " away" : ""}"><text transform="translate(${x0 - 14} ${y0 - size}) rotate(-90)" text-anchor="end">closer to ${esc(b.text)} →</text></g>` +
      `<text class="axis-note" x="${sx(hi) - 4}" y="${sy(hi) + 16}" text-anchor="end">equally close</text>`;
  } else {
    const anchors = pot.map((p, i) => {
      const ang = -Math.PI / 2 + (2 * Math.PI * i) / pot.length;
      return { x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang), a: ang, p };
    });
    place = (res, rank) => {
      if (pot.length === 1) {
        // radial: distance from the word shrinks as similarity grows
        const z = Math.max(res.z[0], 0.5);
        const dist = Math.min(R * 1.8, (R * 1.6) / z);
        const ang = Math.PI / 2 + (rank % 2 ? 1 : -1) * (0.15 + 0.11 * rank);
        return [anchors[0].x + dist * Math.cos(ang), anchors[0].y + dist * Math.sin(ang)];
      }
      const e = res.z.map((z) => Math.exp(TAU * z));
      const sum = e.reduce((acc, v) => acc + v, 0);
      let x = 0, y = 0;
      e.forEach((w, i) => { x += (w / sum) * (anchors[i].x - cx); y += (w / sum) * (anchors[i].y - cy); });
      return [cx + x * 0.88, cy + y * 0.88];
    };
    backdrop =
      (pot.length > 1 ? [0.33, 0.66, 1].map((f) => `<circle class="ring" cx="${cx}" cy="${cy}" r="${R * f}"/>`).join("") : "") +
      (pot.length > 1 ? anchors.map((an) => `<line class="spoke" x1="${cx}" y1="${cy}" x2="${an.x}" y2="${an.y}"/>`).join("") : "") +
      anchors.map((an) => {
        const right = Math.cos(an.a) > 0.25, left = Math.cos(an.a) < -0.25;
        const anchor = right ? "start" : left ? "end" : "middle";
        const dx = right ? 12 : left ? -12 : 0, dy = Math.sin(an.a) > 0.25 ? 24 : Math.sin(an.a) < -0.25 ? -12 : 5;
        return `<g class="anchor${an.p.weight < 0 ? " away" : ""}"><circle cx="${an.x}" cy="${an.y}" r="7"/>` +
          `<text x="${an.x + dx}" y="${an.y + dy}" text-anchor="${anchor}">${esc(an.p.text)}</text></g>`;
      }).join("");
  }

  // keep existing <g> nodes so CSS transitions animate movement
  if (!svg.querySelector(".static")) svg.innerHTML = `<g class="static"></g><g class="cands"></g>`;
  svg.querySelector(".static")!.innerHTML = backdrop;

  const candG = svg.querySelector(".cands")!;
  const existing = new Map<string, SVGGElement>();
  candG.querySelectorAll<SVGGElement>(".cand").forEach((g) => existing.set(g.dataset.key!, g));
  const keep = new Set<string>();
  const taken: [number, number, number, number][] = [];  // label boxes already drawn
  const free = (x: number, y: number, w: number) => !taken.some(([a, b, c, d]) => x < c && x + w > a && y - 12 < d && y + 4 > b);
  // selected first so its label always wins, then by rank
  const order = results.map((r, rank) => ({ r, rank })).sort((a, b) => Number(b.r.index === selected) - Number(a.r.index === selected) || a.rank - b.rank);
  for (const { r: res, rank } of order) {
    const key = String(res.index);
    keep.add(key);
    const [x, y] = place(res, rank);
    let g = existing.get(key);
    if (!g) {
      g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.classList.add("cand");
      g.dataset.key = key;
      g.addEventListener("click", () => onSelect(res.index));
      g.style.transform = `translate(${cx}px, ${cy}px)`;
      candG.appendChild(g);
      void g.getBoundingClientRect();
    }
    const word = space.lex.text[res.index];
    const w = word.length * 7.2 + 8;
    const label = (rank < LABELLED || res.index === selected) && free(x + 7, y, w);
    if (label) taken.push([x + 7, y - 12, x + 7 + w, y + 4]);
    g.innerHTML = `<title>${esc(word)}</title><circle r="${rank < 3 ? 4.5 : 3}"/>` + (label ? `<text x="7" y="4">${esc(word)}</text>` : "");
    g.classList.toggle("sel", res.index === selected);
    g.style.transform = `translate(${x}px, ${y}px)`;
  }
  existing.forEach((g, k) => { if (!keep.has(k)) g.remove(); });
}

export function renderEquation(el: HTMLElement, space: Space, res: Result | undefined, s: Settings): void {
  if (!res) { el.innerHTML = ""; return; }
  const b = res.breakdown;
  const axisLabel = new Map(space.meta.axes.map((a) => [a.key, a]));
  const bar = (v: number) => `<span class="bar${v < 0 ? " neg" : ""}" style="width:${Math.min(60, Math.abs(v) * 14)}px"></span>`;
  const rows: string[] = [];
  const pos = b.perItem.filter((p) => p.weight > 0);
  const neg = b.perItem.filter((p) => p.weight < 0);
  for (const p of pos) rows.push(`<tr><td>close to <em>${esc(p.text)}</em> <small>(weight ${p.weight.toFixed(1)})</small>${bar(p.z)}</td><td class="n">${p.z.toFixed(2)}σ</td></tr>`);
  if (pos.length) {
    const how = s.beta < 0.05 ? "weighted mean" : `soft minimum, β = ${s.beta.toFixed(1)}`;
    rows.push(`<tr><td><strong>fit</strong> <small>(${how})</small></td><td class="n">${fmt(b.fit)}</td></tr>`);
  }
  for (const p of neg) {
    const pen = -p.weight * Math.max(0, p.z);
    rows.push(`<tr><td>away from <em>${esc(p.text)}</em> <small>(${p.z.toFixed(2)}σ × ${(-p.weight).toFixed(1)})</small>${bar(-pen)}</td><td class="n">${fmt(-pen)}</td></tr>`);
  }
  for (const d of b.dials) {
    const a = axisLabel.get(d.key)!;
    const raw = space.lex.axes[d.key][res.index];
    rows.push(`<tr><td>${esc(a.label.toLowerCase())} <small>(word at ${raw.toFixed(2)}σ × dial ${d.gain.toFixed(2)})</small>${bar(d.value)}</td><td class="n">${fmt(d.value)}</td></tr>`);
  }
  if (s.softness) {
    const known = space.lex.hasSound[res.index];
    rows.push(`<tr><td>sound <small>${known ? `(softness ${space.lex.soft[res.index].toFixed(2)}σ × ${s.softness.toFixed(2)})` : "(pronunciation unknown)"}</small>${bar(b.sound)}</td><td class="n">${fmt(b.sound)}</td></tr>`);
  }
  if (s.rarity && b.rarity > 0) rows.push(`<tr><td>rarity <small>(frequency ${space.lex.zipf[res.index].toFixed(1)} on the Zipf scale)</small>${bar(-b.rarity)}</td><td class="n">${fmt(-b.rarity)}</td></tr>`);
  rows.push(`<tr class="total"><td>score</td><td class="n">${b.total.toFixed(2)}</td></tr>`);
  el.innerHTML =
    `<h3>${esc(space.lex.text[res.index])}</h3><table>${rows.join("")}</table>` +
    (b.perItem.length
      ? `<p class="note">Only the ${SHORTLIST} entries that best fit your words are considered; the dials, sound and rarity then re-rank those.</p>`
      : `<p class="note">With no words added, the dials rank the whole vocabulary.</p>`);
}
