/**
 * App shell: state, persistence, and wiring the controls to the engine.
 * The maths lives in engine.ts; the "geometry" view in hood.ts.
 */

import { loadSpace, suggest, type Space } from "./data";
import { DEFAULT_BETA, rank, vectorFor, zSimilarities, type PotItem, type Result, type Settings } from "./engine";
import { drawMap, renderEquation } from "./hood";
import { respell } from "./respell";
import "./styles.css";

const DATA_URL = "./data";
const STORE_KEY = "contour.v1";
const PICKS_KEY = "contour.picks.v1";

/**
 * Sound controls (softness, syllables, starts like, rhymes with) are switched
 * off for now. Set to true to bring the Sound section back; nothing else changes.
 */
const SOUND_CONTROLS = false;

const defaults = (space: Space): Settings => ({
  beta: DEFAULT_BETA,
  dials: Object.fromEntries(space.meta.axes.map((a) => [a.key, 0])),
  softness: 0, syllables: 0, startsLike: "", rhymesWith: "",
  rarity: 0.5, phrases: true, hideOffensive: true,
});

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const store = {
  get<T>(key: string, fallback: T): T { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(key: string, v: unknown) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

let space: Space;
let settings: Settings;
let pot: PotItem[] = [];
const zCache = new Map<string, Float32Array>();
let results: Result[] = [];
let selected = -1;
let hoodOn = false;
let nextId = 0;

/* ---------- pot ---------- */

function addToPot(text: string, weight = 1): string | null {
  const clean = text.trim().toLowerCase().replace(/\s+/g, " ");
  if (!clean) return null;
  if (pot.some((p) => p.text === clean)) return `“${clean}” is already in your list.`;
  const v = vectorFor(space, clean);
  if (!v) return `No word in “${clean}” is in the vocabulary. Check the spelling, or try a related word.`;
  pot.push({ id: String(nextId++), text: clean, weight, ...v });
  return null;
}

function zFor(item: PotItem): Float32Array {
  let z = zCache.get(item.text);
  if (!z) { z = zSimilarities(space, item.vector); zCache.set(item.text, z); }
  return z;
}

function renderPot(): void {
  $("pot-list").innerHTML = pot.map((p) => `
    <li class="${p.weight < 0 ? "away" : ""}" data-id="${p.id}">
      <span class="word">${esc(p.text)}</span>
      <span class="amount">${p.weight < 0 ? "steer away" : p.weight === 0 ? "off" : "pull toward"} ${Math.abs(p.weight).toFixed(1)}</span>
      <button type="button" aria-label="Remove ${esc(p.text)}">×</button>
      <input type="range" min="-2" max="2" step="0.1" value="${p.weight}" aria-label="Weight of ${esc(p.text)}" />
      ${p.approx ? `<span class="approx">Not in the vocabulary as a whole, so approximated from: ${p.parts.map(esc).join(", ")}</span>` : ""}
    </li>`).join("");
}

/* ---------- ranking ---------- */

function update(): void {
  const t0 = performance.now();
  results = rank(space, pot, pot.map(zFor), settings, 40);
  const ms = performance.now() - t0;
  if (!results.some((r) => r.index === selected)) selected = results[0]?.index ?? -1;
  $("timing").textContent = results.length ? `ranked ${space.count.toLocaleString()} entries in ${Math.max(1, Math.round(ms))} ms` : "";
  $("empty").hidden = results.length > 0;
  renderResults();
  renderHood();
  save();
}

function renderResults(): void {
  const top = results[0]?.score ?? 1;
  const bottom = results[results.length - 1]?.score ?? 0;
  $("results").innerHTML = results.map((r) => {
    const { lex } = space;
    const sound = lex.phones[r.index] ? `${lex.guessed[r.index] ? "~" : ""}${respell(lex.phones[r.index])}` : "";
    const width = top > bottom ? 12 + (88 * (r.score - bottom)) / (top - bottom) : 100;
    return `<li class="${r.index === selected ? "selected" : ""}" data-row="${r.index}">
      <button type="button" class="row" aria-pressed="${r.index === selected}">
        <span><span class="w">${esc(lex.text[r.index])}</span><span class="ph">${esc(sound)}</span></span>
        <span class="meter" aria-hidden="true"><i style="width:${width.toFixed(1)}%"></i></span>
        <span class="score">${r.score.toFixed(1)}σ</span>
      </button>
      <button type="button" class="this-one">This one</button>
    </li>`;
  }).join("");
}

function renderHood(): void {
  if (!hoodOn) return;
  const svg = document.querySelector<SVGSVGElement>("#map svg")!;
  $("map-help").textContent =
    pot.length === 0 ? "Add a word to see where the suggestions sit relative to it."
    : pot.length === 1 ? "Suggestions closer to your word sit nearer to it."
    : pot.length === 2 ? "Each suggestion is plotted by how close it is to each of your two words (in σ). Words near the dashed line are equally close to both."
    : "Your words sit on the ring. Each suggestion is pulled toward the words it is closest to; one equally close to all of them sits in the middle.";
  drawMap(svg, space, pot, results.slice(0, 30), selected, (r) => { selected = r; renderResults(); renderHood(); });
  renderEquation($("equation"), space, results.find((r) => r.index === selected), settings);
}

/* ---------- controls ---------- */

function renderDials(): void {
  $("dials").innerHTML = space.meta.axes.map((a) => `
    <div class="dial" data-key="${a.key}">
      <label for="dial-${a.key}">${esc(a.label)} <output></output></label>
      <input type="range" id="dial-${a.key}" min="-1" max="1" step="0.05" title="${esc(a.description)}" />
      <div class="ends"><span>${esc(a.negative)}</span><span>${esc(a.positive)}</span></div>
    </div>`).join("");
}

function syncControls(): void {
  ($("beta") as HTMLInputElement).value = String(settings.beta);
  for (const a of space.meta.axes) {
    const v = settings.dials[a.key] ?? 0;
    const wrap = document.querySelector<HTMLElement>(`.dial[data-key="${a.key}"]`)!;
    wrap.querySelector("input")!.value = String(v);
    wrap.querySelector("output")!.textContent = v ? (v > 0 ? `more ${a.positive}` : `more ${a.negative}`) : "off";
    wrap.classList.toggle("active", v !== 0);
  }
  ($("softness") as HTMLInputElement).value = String(settings.softness);
  ($("syllables") as HTMLSelectElement).value = String(settings.syllables);
  ($("starts-like") as HTMLInputElement).value = settings.startsLike;
  ($("rhymes-with") as HTMLInputElement).value = settings.rhymesWith;
  ($("rarity") as HTMLInputElement).value = String(settings.rarity);
  ($("phrases") as HTMLInputElement).checked = settings.phrases;
  ($("hide-offensive") as HTMLInputElement).checked = settings.hideOffensive;
}

/** Settings as loaded or restored, with the sound filters neutral while they are switched off. */
function withSoundPolicy(st: Settings): Settings {
  return SOUND_CONTROLS ? st : { ...st, softness: 0, syllables: 0, startsLike: "", rhymesWith: "" };
}

function save(): void {
  store.set(STORE_KEY, { pot: pot.map((p) => [p.text, p.weight]), settings, hood: hoodOn });
}

/* ---------- picks (feedback log) ---------- */

interface Pick { at: string; word: string; rank: number; score: number; pot: [string, number][]; settings: Settings; model: string }

function recordPick(row: number): void {
  const res = results.findIndex((r) => r.index === row);
  const picks = store.get<Pick[]>(PICKS_KEY, []);
  picks.push({
    at: new Date().toISOString(), word: space.lex.text[row], rank: res + 1,
    score: Number(results[res].score.toFixed(3)), pot: pot.map((p) => [p.text, p.weight]),
    settings: structuredClone(settings), model: space.meta.model,
  });
  store.set(PICKS_KEY, picks);
  $("picks-count").textContent = String(picks.length);
  toast(`Saved “${space.lex.text[row]}” to your picks`);
}

function toast(msg: string): void {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- autocomplete ---------- */

function wireAddInput(): void {
  const input = $("add-input") as HTMLInputElement;
  const list = $("suggest-list");
  let items: string[] = [];
  let active = -1;
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; };
  const show = () => {
    items = suggest(space, input.value);
    active = -1;
    list.innerHTML = items.map((w, i) => `<li role="option" id="sug-${i}" aria-selected="false">${esc(w)}</li>`).join("");
    list.hidden = !items.length;
    input.setAttribute("aria-expanded", String(!!items.length));
  };
  const commit = (text: string) => {
    const err = addToPot(text);
    $("add-error").hidden = !err;
    $("add-error").textContent = err ?? "";
    if (!err) { input.value = ""; renderPot(); update(); }
    close();
  };
  input.addEventListener("input", show);
  input.addEventListener("keydown", (e) => {
    if (list.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      list.querySelectorAll("li").forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
      input.setAttribute("aria-activedescendant", `sug-${active}`);
    } else if (e.key === "Escape") close();
  });
  list.addEventListener("mousedown", (e) => {
    const li = (e.target as HTMLElement).closest("li");
    if (li) { e.preventDefault(); commit(li.textContent ?? ""); }
  });
  input.addEventListener("blur", () => setTimeout(close, 100));
  $("add-form").addEventListener("submit", (e) => {
    e.preventDefault();
    commit(active >= 0 ? items[active] : input.value);
  });
}

/* ---------- boot ---------- */

function wire(): void {
  wireAddInput();

  $("pot-list").addEventListener("input", (e) => {
    const li = (e.target as HTMLElement).closest("li")!;
    const item = pot.find((p) => p.id === li.dataset.id)!;
    item.weight = Number((e.target as HTMLInputElement).value);
    li.classList.toggle("away", item.weight < 0);
    li.querySelector(".amount")!.textContent = `${item.weight < 0 ? "steer away" : item.weight === 0 ? "off" : "pull toward"} ${Math.abs(item.weight).toFixed(1)}`;
    update();
  });
  $("pot-list").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest("button");
    if (!btn) return;
    const id = btn.closest("li")!.dataset.id;
    pot = pot.filter((p) => p.id !== id);
    renderPot(); update();
  });

  const num = (id: string, set: (v: number) => void) =>
    $(id).addEventListener("input", (e) => { set(Number((e.target as HTMLInputElement).value)); syncControls(); update(); });
  num("beta", (v) => (settings.beta = v));
  num("softness", (v) => (settings.softness = v));
  num("rarity", (v) => (settings.rarity = v));
  $("syllables").addEventListener("change", (e) => { settings.syllables = Number((e.target as HTMLSelectElement).value); update(); });
  $("dials").addEventListener("input", (e) => {
    const key = (e.target as HTMLElement).closest<HTMLElement>(".dial")!.dataset.key!;
    settings.dials[key] = Number((e.target as HTMLInputElement).value);
    syncControls(); update();
  });
  const text = (id: string, set: (v: string) => void) =>
    $(id).addEventListener("input", (e) => { set((e.target as HTMLInputElement).value.trim().toLowerCase()); update(); });
  text("starts-like", (v) => (settings.startsLike = v));
  text("rhymes-with", (v) => (settings.rhymesWith = v));
  $("phrases").addEventListener("change", (e) => { settings.phrases = (e.target as HTMLInputElement).checked; update(); });
  $("hide-offensive").addEventListener("change", (e) => { settings.hideOffensive = (e.target as HTMLInputElement).checked; update(); });

  $("results").addEventListener("click", (e) => {
    const li = (e.target as HTMLElement).closest<HTMLElement>("li");
    if (!li) return;
    const row = Number(li.dataset.row);
    if ((e.target as HTMLElement).closest(".this-one")) recordPick(row);
    else { selected = row; renderResults(); renderHood(); }
  });

  const hood = $("hood-toggle") as HTMLInputElement;
  const setHood = (on: boolean) => {
    hoodOn = on; hood.checked = on;
    $("hood").hidden = !on;
    $("app").classList.toggle("with-hood", on);
    renderHood(); save();
  };
  hood.addEventListener("change", () => setHood(hood.checked));
  new ResizeObserver(() => renderHood()).observe($("map"));

  const dialog = $("picks-dialog") as HTMLDialogElement;
  $("picks-button").addEventListener("click", () => {
    const picks = store.get<Pick[]>(PICKS_KEY, []);
    $("picks-list").innerHTML = picks.length
      ? picks.slice().reverse().map((p) => `<li><span class="w">${esc(p.word)}</span> — rank ${p.rank}, from ${p.pot.map(([t]) => esc(t)).join(" + ") || "dials only"}</li>`).join("")
      : "<li>No picks yet. Mark a suggestion with “This one”.</li>";
    try { dialog.showModal(); } catch { dialog.setAttribute("open", ""); }
  });
  $("copy-picks").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(store.get(PICKS_KEY, []), null, 1));
      toast("Copied your picks as JSON");
    } catch { toast("Copying isn’t allowed here. Your picks are still saved in this browser."); }
  });
  $("clear-picks").addEventListener("click", () => {
    store.set(PICKS_KEY, []);
    $("picks-count").textContent = "0";
    $("picks-list").innerHTML = "<li>No picks yet.</li>";
  });

  setHood(store.get<{ hood?: boolean }>(STORE_KEY, {}).hood ?? false);
}

async function boot(): Promise<void> {
  const loading = $("loading");
  try {
    space = await loadSpace(DATA_URL, (mb) => { loading.textContent = `Loading the vocabulary… ${mb.toFixed(0)} MB`; });
  } catch (err) {
    loading.textContent = `The vocabulary didn’t load: ${(err as Error).message}. Reload the page to try again.`;
    return;
  }
  loading.hidden = true;
  $("app").setAttribute("aria-busy", "false");
  if (space.meta.stand_in) {
    const note = $("stand-in");
    note.hidden = false;
    note.innerHTML = `<strong>Preview space:</strong> ${esc(space.meta.model)} static word vectors stand in for the planned Qwen3-Embedding model, which this preview can’t load. The interface and maths are the same; the suggestions will change when the space does.`;
  }

  renderDials();
  const saved = store.get<{ pot?: [string, number][]; settings?: Settings }>(STORE_KEY, {});
  $("picks-count").textContent = String(store.get<Pick[]>(PICKS_KEY, []).length);
  wire();
  settings = saved.settings
    ? withSoundPolicy({ ...defaults(space), ...saved.settings, dials: { ...defaults(space).dials, ...saved.settings.dials } })
    : defaults(space);
  for (const [t, w] of saved.pot ?? []) addToPot(t, w);
  $("sound-block").hidden = !SOUND_CONTROLS;
  renderPot(); syncControls(); update();
}

boot();
