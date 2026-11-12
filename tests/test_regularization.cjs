"use strict";

// Deterministic checks for the dropout and end-to-end CNN lessons (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const dir = path.join(__dirname, "..", "docs", "pregrado", "animations");
const S = require(path.join(dir, "spatial.js"));
const O = require(path.join(dir, "operations.js"));
const R = require(path.join(dir, "regularization.js"));

const ACTIVATIONS = [2, 1, 3, 2, 4, 1];
// Hand-computed inverted dropout, p = 0.5: kept values double, dropped values become 0.
const PASS_OUTPUTS = [
  [4, 0, 6, 4, 0, 0],
  [0, 2, 6, 0, 8, 2],
  [4, 2, 0, 0, 8, 0],
  [4, 0, 6, 4, 8, 0],
];

// House on a 6×6 canvas (extra zero row and column), hand-computed with stride 1, no padding.
const CANVAS = [
  [0, 0, 1, 0, 0, 0],
  [0, 1, 0, 1, 0, 0],
  [1, 1, 1, 1, 1, 0],
  [0, 1, 0, 1, 0, 0],
  [0, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0],
];
const CONV = [
  [
    [2, 2, 2, 1],
    [3, 1, 3, 1],
    [3, 2, 3, 1],
    [2, 1, 2, 0],
  ],
  [
    [-2, -1, -2, -1],
    [2, 1, 2, 1],
    [-1, -1, -1, 0],
    [2, 3, 2, 1],
  ],
];
const RELU = [
  CONV[0],
  [
    [0, 0, 0, 0],
    [2, 1, 2, 1],
    [0, 0, 0, 0],
    [2, 3, 2, 1],
  ],
];
const POOLED = [
  [
    [3, 3],
    [3, 3],
  ],
  [
    [2, 2],
    [3, 2],
  ],
];
const FLAT = [3, 3, 3, 3, 2, 2, 3, 2];

function cells(svg, prefix) {
  const re = new RegExp(`<rect[^>]*data-cell="${prefix}-(\\d+)-(\\d+)"[^>]*>`, "g");
  const out = {};
  for (const m of svg.matchAll(re)) {
    const tag = m[0];
    out[`${m[1]}-${m[2]}`] = {
      value: Number(/data-value="(-?[\d.]+)"/.exec(tag)[1]),
      dropped: /data-dropped="1"/.test(tag),
      clamped: /data-clamped="1"/.test(tag),
    };
  }
  return out;
}
const values = (svg, prefix) => Object.fromEntries(Object.entries(cells(svg, prefix)).map(([k, c]) => [k, c.value]));
const asCells = (m) => Object.fromEntries(m.flatMap((row, r) => row.map((v, c) => [`${r}-${c}`, v])));
const rowCells = (v) => asCells([v]);
const frames = (kind) => Array.from({ length: R.LESSONS[kind].frameCount }, (_, f) => R.LESSONS[kind].render(f));
const deepFreeze = (m) => Object.freeze(m.map((x) => (Array.isArray(x) ? deepFreeze(x) : x)));
const stage = (svg) => (/data-stage="([a-z]+)"/.exec(svg) || [])[1] || null;

// ---------- Pure dropout ----------

test("lessons register in the shared SpatialLesson registry", () => {
  for (const kind of ["dropout", "pipeline"]) {
    assert.equal(S.LESSONS[`cnn-${kind}`], R.LESSONS[kind]);
    assert.ok(R.LESSONS[kind].frameCount > 1);
  }
});

test("inverted dropout: kept values divided by (1 - p), dropped values are 0, shape unchanged", () => {
  const out = R.dropout(ACTIVATIONS, { rate: 0.5, mask: [1, 0, 1, 1, 0, 0], training: true });
  assert.deepEqual(out, [4, 0, 6, 4, 0, 0]);
  assert.equal(out.length, ACTIVATIONS.length);
  assert.ok(out.every((v) => !Object.is(v, -0)), "dropped values are +0");
  assert.deepEqual(R.dropout([0, 5], { rate: 0.5, mask: [1, 1] }), [0, 10], "a kept zero stays 0; training is default");
  assert.deepEqual(R.dropout([-1, 3], { rate: 0.25, mask: [1, 0], training: true }), [-4 / 3, 0]);
  const grid = R.dropout([[1, 2], [3, 4]], { rate: 0.5, mask: [[1, 0], [0, 1]], training: true });
  assert.deepEqual(grid, [[2, 0], [0, 8]], "2D input keeps its 2×2 shape");
});

test("rate boundaries: 0 drops nothing, values below 1 rescale, 1 and invalid options are rejected", () => {
  assert.deepEqual(R.dropout(ACTIVATIONS, { rate: 0, mask: [0, 0, 0, 0, 0, 0], training: true }), ACTIVATIONS);
  assert.deepEqual(R.dropout([1], { rate: 0.9, mask: [1], training: true })[0], 1 / (1 - 0.9));
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: 1, mask: ACTIVATIONS.map(() => 0), training: true }), /rate/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: -0.1, mask: ACTIVATIONS.map(() => 1) }), /rate/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: NaN, mask: ACTIVATIONS.map(() => 1) }), /rate/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: 0.5, training: true }), /mask/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: 0.5, mask: [1, 0], training: true }), /mask/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: 0.5, mask: [1, 0, 2, 1, 1, 1], training: true }), /mask/);
  assert.throws(() => R.dropout([1, Infinity], { rate: 0.5, mask: [1, 1] }), /finite/);
  assert.throws(() => R.dropout([], { rate: 0.5, mask: [] }), /finite/);
  assert.throws(() => R.dropout(ACTIVATIONS, { rate: 0.5, mask: [1, 1, 1, 1, 1, 1], training: "yes" }), /training/);
});

test("inference returns an identical copy regardless of the mask, with no extra rescale", () => {
  const input = deepFreeze([[2, 1], [3, 0]]);
  for (const mask of [undefined, [[0, 0], [0, 0]], [[1, 0], [0, 1]]]) {
    const out = R.dropout(input, { rate: 0.5, mask, training: false });
    assert.deepEqual(out, [[2, 1], [3, 0]]);
    assert.notEqual(out, input, "a copy, never the same array");
    assert.notEqual(out[0], input[0]);
  }
});

test("dropout never mutates its inputs", () => {
  const input = deepFreeze(ACTIVATIONS.slice());
  const mask = deepFreeze([1, 0, 1, 0, 1, 0]);
  assert.doesNotThrow(() => R.dropout(input, { rate: 0.5, mask, training: true }));
  assert.deepEqual(input, ACTIVATIONS);
});

test("dropoutData: four distinct illustrative masks, every activation dropped and kept at least once", () => {
  const d = R.dropoutData();
  assert.deepEqual(d.activations, ACTIVATIONS);
  assert.equal(d.rate, 0.5);
  assert.equal(d.masks.length, 4);
  assert.equal(new Set(d.masks.map((m) => m.join(""))).size, 4, "each pass uses a new mask");
  const dropCounts = d.masks.map((m) => m.filter((x) => x === 0).length);
  assert.ok(new Set(dropCounts).size > 1, "drop count varies: not exactly half every time");
  ACTIVATIONS.forEach((_, i) => {
    assert.ok(d.masks.some((m) => m[i] === 0), `activation ${i} is dropped in some pass`);
    assert.ok(d.masks.some((m) => m[i] === 1), `activation ${i} comes back in some pass`);
  });
  assert.deepEqual(d.passes.map((p) => p.output), PASS_OUTPUTS);
  assert.deepEqual(d.inference, ACTIVATIONS);
});

// ---------- Dropout frames ----------

test("every frame of both lessons is a 600x440 SVG with a spoken label", () => {
  for (const kind of Object.keys(R.LESSONS)) {
    frames(kind).forEach((out) => {
      assert.match(out.svg, /^<svg[^>]*class="spatial-svg regularization-svg"[^>]*viewBox="0 0 600 440"/);
      assert.ok(out.svg.endsWith("</svg>"));
      assert.ok(out.label.length > 20);
    });
  }
});

test("dropout frames: each training pass shows its mask, the survivors and the rescaled values", () => {
  const all = frames("dropout");
  const d = R.dropoutData();
  assert.equal(all.length, 10);
  all.forEach(({ svg }) => assert.deepEqual(values(svg, "act"), rowCells(ACTIVATIONS), "input stays the same"));
  assert.deepEqual(values(all[0].svg, "out"), rowCells(ACTIVATIONS), "before training: everything flows");
  assert.deepEqual(values(all[0].svg, "mask"), {});
  for (let k = 0; k < 4; k += 1) {
    const mask = d.masks[k];
    const kept = ACTIVATIONS.map((a, i) => a * mask[i]);
    const [look, scaled] = [all[1 + 2 * k], all[2 + 2 * k]];
    [look, scaled].forEach(({ svg, label }) => {
      assert.deepEqual(values(svg, "mask"), rowCells(mask));
      const out = cells(svg, "out");
      mask.forEach((m, i) => assert.equal(out[`0-${i}`].dropped, m === 0, `pass ${k + 1}, cell ${i}`));
      assert.match(svg, new RegExp(`Entrenamiento · pasada ${k + 1}/4`));
      assert.match(label, new RegExp(`pasada ${k + 1}/4`));
      for (let j = 0; j <= k; j += 1) assert.deepEqual(values(svg, `hist${j}`), rowCells(d.masks[j]), "mask history");
      assert.deepEqual(values(svg, `hist${k + 1}`), {}, "future masks are not shown");
    });
    assert.deepEqual(values(look.svg, "out"), rowCells(kept));
    assert.deepEqual(values(scaled.svg, "out"), rowCells(PASS_OUTPUTS[k]));
    assert.match(scaled.label, /÷ 0\.5|\/ 0\.5/);
  }
  // Numeric example of the first activation (2): 4 when kept, 0 when dropped.
  assert.match(all[2].label, /2 → 4/);
  assert.match(all[4].label, /2 → 0/);
});

test("dropout inference frame: no masks, original values, no extra rescale, honest wording", () => {
  const { svg, label } = frames("dropout")[9];
  assert.deepEqual(values(svg, "out"), rowCells(ACTIVATIONS));
  assert.deepEqual(values(svg, "mask"), {});
  assert.ok(Object.values(cells(svg, "out")).every((c) => !c.dropped));
  assert.match(svg, /Inferencia/);
  assert.match(label, /Inferencia/);
  assert.match(label, /2 → 2/);
  const all = frames("dropout").map((f) => f.svg + f.label).join(" ");
  assert.match(all, /ilustrativ/);
  assert.match(all, /al azar/);
  assert.match(all, /no garantiza/);
  assert.match(all, /6 → 6/, "shape is preserved");
  assert.doesNotMatch(all, /exactamente la mitad|elimina(?:n)? (?:la )?neurona(?:s)? para siempre|permanentemente se/i);
});

// ---------- End-to-end pipeline ----------

test("pipelineData: canvas, two filters, conv, ReLU, pooling, flatten and dense scores from the helpers", () => {
  const p = R.pipelineData();
  assert.deepEqual(p.image, CANVAS);
  assert.deepEqual(p.image, S.padInto(S.HOUSE, 6, 0, 0));
  assert.equal(p.filters.length, 2);
  assert.equal(p.filters[0].kernel, S.KERNELS.vertical, "canonical kernel reused, not copied");
  assert.ok(p.filters[1].kernel.flat().some((w) => w < 0), "second filter has negative weights");
  p.filters.forEach((f, i) => assert.deepEqual(p.conv[i], O.convolve(CANVAS, f.kernel)));
  assert.deepEqual(p.conv, CONV);
  assert.deepEqual(p.relu, RELU);
  assert.deepEqual(p.pooled, POOLED);
  p.relu.forEach((m, i) => assert.deepEqual(p.pooled[i], O.pool2d(m, { size: 2, stride: 2, mode: "max" })));
  assert.deepEqual(p.flat, FLAT);
  assert.equal(p.weights.length, 8);
  p.weights.forEach((row) => assert.equal(row.length, 2));
  const manual = [0, 1].map((j) => FLAT.reduce((s, x, i) => s + x * p.weights[i][j], 0));
  assert.deepEqual(p.scores, manual);
  assert.deepEqual(p.shapes, ["6×6", "4×4×2", "4×4×2", "2×2×2", "8", "2"]);
  const clampedCount = CONV[1].flat().filter((v) => v < 0).length;
  assert.equal(clampedCount, 7);
  assert.equal(p.relu.flat(2).filter((v) => v < 0).length, 0);
});

test("pipeline frames follow image → conv → ReLU → pool → flatten → scores without hiding earlier maps", () => {
  const all = frames("pipeline");
  const p = R.pipelineData();
  const order = ["image", "conv", "relu", "pool", "flatten", "dense"];
  assert.deepEqual(all.slice(0, 6).map((f) => stage(f.svg)), order);
  all.forEach(({ svg }, f) => {
    assert.deepEqual(values(svg, "img"), asCells(CANVAS), "the image is always visible");
    if (f >= 1) [0, 1].forEach((i) => assert.deepEqual(values(svg, `conv${i}`), asCells(CONV[i])));
    if (f >= 2) [0, 1].forEach((i) => assert.deepEqual(values(svg, `relu${i}`), asCells(RELU[i])));
    if (f >= 3) [0, 1].forEach((i) => assert.deepEqual(values(svg, `pool${i}`), asCells(POOLED[i])));
    if (f >= 4) assert.deepEqual(values(svg, "flat"), Object.fromEntries(FLAT.map((v, i) => [`${i}-0`, v])));
    if (f >= 5) assert.deepEqual(values(svg, "score"), Object.fromEntries(p.scores.map((v, i) => [`${i}-0`, v])));
    if (f < 5) assert.deepEqual(values(svg, "score"), {}, "no score before the dense step");
    assert.match(svg, /parámetros ilustrativos, no entrenados/);
    assert.doesNotMatch(svg, /probabilidad|softmax\(/i);
  });
  assert.deepEqual(values(all[0].svg, "conv0"), {});
  // ReLU frame marks exactly the negative responses of the second map as clamped.
  const relu = cells(all[2].svg, "relu1");
  Object.entries(relu).forEach(([k, c]) => {
    const [r, col] = k.split("-").map(Number);
    assert.equal(c.clamped, CONV[1][r][col] < 0, `relu1 ${k}`);
  });
  assert.match(all[2].label, /−2 → 0|-2 → 0/);
  assert.match(all[3].label, /max/);
  assert.match(all[4].label, /3, 3, 3, 3, 2, 2, 3, 2/);
  assert.match(all[5].label, new RegExp(`${p.scores[0]}`));
  assert.match(all[5].label, new RegExp(`${p.scores[1]}`));
  const last = all[all.length - 1];
  assert.match(last.svg, /6×6 → 4×4×2 → 4×4×2 → 2×2×2 → 8 → 2/);
  assert.match(last.svg, /[Dd]ropout/);
  assert.match(last.svg, /solo en entrenamiento/);
});

// ---------- Mount with a minimal fake DOM ----------

function fakeDom({ reduced = false } = {}) {
  const timers = new Map();
  let id = 0;
  const sched = {
    timers,
    setInterval(fn) {
      id += 1;
      timers.set(id, fn);
      return id;
    },
    clearInterval(t) {
      timers.delete(t);
    },
    tick() {
      for (const fn of [...timers.values()]) fn();
    },
  };
  const observers = [];
  const listeners = {};
  const doc = {
    readyState: "complete",
    hidden: false,
    mounts: [],
    addEventListener: (t, fn) => (listeners[t] ||= []).push(fn),
    removeEventListener() {},
    fire: (t) => (listeners[t] || []).forEach((fn) => fn()),
  };
  class El {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.attrs = {};
      this.listeners = {};
      this.classes = new Set();
      this.classList = {
        add: (...c) => c.forEach((x) => this.classes.add(x)),
        remove: (...c) => c.forEach((x) => this.classes.delete(x)),
        contains: (c) => this.classes.has(c),
      };
      this.textContent = "";
      this.innerHTML = "";
      this.parentNode = null;
      this.ownerDocument = doc;
    }
    appendChild(c) {
      c.parentNode = this;
      return c;
    }
    setAttribute(k, v) {
      this.attrs[k] = String(v);
    }
    getAttribute(k) {
      return k in this.attrs ? this.attrs[k] : null;
    }
    addEventListener(t, fn) {
      (this.listeners[t] ||= []).push(fn);
    }
    dispatch(t, ev) {
      (this.listeners[t] || []).forEach((fn) => fn(ev));
    }
    closest(sel) {
      let n = this;
      while (n && !n.classList.contains(sel.slice(1))) n = n.parentNode;
      return n;
    }
    blur() {}
  }
  class MO {
    constructor(cb) {
      this.cb = cb;
      observers.push(this);
    }
    observe() {}
    disconnect() {}
  }
  doc.defaultView = {
    setInterval: sched.setInterval,
    clearInterval: sched.clearInterval,
    matchMedia: () => ({ matches: reduced, addEventListener() {} }),
    MutationObserver: MO,
  };
  doc.createElement = (t) => new El(t);
  doc.querySelectorAll = (sel) => (sel === "[data-regularization-lesson]" ? doc.mounts : []);
  const notify = () => observers.forEach((o) => o.cb([]));
  const lessonOnSlide = (kind, active) => {
    const slide = new El("section");
    slide.classList.add("slide");
    if (active) slide.classList.add("is-active");
    const el = new El("div");
    el.setAttribute("data-regularization-lesson", kind);
    slide.appendChild(el);
    doc.mounts.push(el);
    return { slide, el };
  };
  return { doc, sched, notify, lessonOnSlide };
}

const key = (k, target) => ({
  key: k,
  target,
  stopped: false,
  stopPropagation() {
    this.stopped = true;
  },
  preventDefault() {},
});

test("autoMount builds both lessons and plays only on the active, visible slide", () => {
  const dom = fakeDom();
  const slides = ["dropout", "pipeline"].map((kind) => dom.lessonOnSlide(kind, false));
  dom.lessonOnSlide("unknown", false);
  const insts = R.autoMount(dom.doc);
  assert.equal(insts.length, 2, "unknown kinds are skipped");
  insts.forEach((inst) => {
    assert.ok(inst.root.classList.contains("regularization-lesson"));
    assert.match(inst.stage.innerHTML, /^<svg[^>]*viewBox="0 0 600 440"/);
    assert.equal(inst.playButton.textContent, "Pausar");
    assert.equal(inst.nextButton.textContent, "Paso siguiente");
  });
  assert.equal(dom.sched.timers.size, 0, "inactive slides: no timer");
  assert.equal(R.autoMount(dom.doc)[0], insts[0], "mount is idempotent");
  slides[1].slide.classList.add("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 1, "only the active slide plays");
  dom.sched.tick();
  assert.equal(insts[1].player.state.frame, 1);
  assert.equal(insts[0].player.state.frame, 0, "state stays isolated per widget");
  dom.doc.hidden = true;
  dom.doc.fire("visibilitychange");
  assert.equal(dom.sched.timers.size, 0, "hidden tab clears the timer");
  insts.forEach((inst) => inst.destroy());
  assert.throws(() => R.mount(dom.lessonOnSlide("nope", true).el), /Unknown regularization lesson/);
});

test("Play/Pause, Step and scoped deck keys; reduced motion starts paused", () => {
  const dom = fakeDom();
  const { el } = dom.lessonOnSlide("dropout", true);
  const inst = R.mount(el);
  assert.equal(dom.sched.timers.size, 1);
  inst.playButton.dispatch("click", { detail: 1 });
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 1 });
  assert.equal(inst.player.state.frame, 1);
  const end = key("End", el);
  el.dispatch("keydown", end);
  assert.equal(end.stopped, true);
  assert.match(inst.label.textContent, /Inferencia/);
  inst.destroy();

  const calm = fakeDom({ reduced: true });
  const { el: el2 } = calm.lessonOnSlide("pipeline", true);
  const inst2 = R.mount(el2);
  assert.equal(calm.sched.timers.size, 0);
  assert.equal(inst2.playButton.textContent, "Reproducir");
  inst2.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst2.player.state.frame, 1);
  inst2.destroy();
});
