"use strict";

// Deterministic checks for the stride, padding and pooling lessons (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const dir = path.join(__dirname, "..", "docs", "pregrado", "animations");
const S = require(path.join(dir, "spatial.js"));
const O = require(path.join(dir, "operations.js"));

const HOUSE = [
  [0, 0, 1, 0, 0],
  [0, 1, 0, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 1, 1, 0],
];
const VALID_MAP = [
  [2, 2, 2],
  [3, 1, 3],
  [3, 2, 3],
];
// Hand-computed: vertical kernel over the house with a 1-pixel zero border.
const PADDED_MAP = [
  [0, 1, 1, 1, 0],
  [1, 2, 2, 2, 1],
  [1, 3, 1, 3, 1],
  [1, 3, 2, 3, 1],
  [0, 2, 1, 2, 0],
];
const POOL_INPUT = [
  [1, 3, 0, 2],
  [2, 9, 1, 1],
  [0, 1, 4, 4],
  [2, 3, 4, 4],
];
const MAX_MAP = [
  [9, 2],
  [3, 4],
];
const AVG_MAP = [
  [3.75, 1],
  [1.5, 4],
];

function cells(svg, prefix) {
  const re = new RegExp(`<rect[^>]*data-cell="${prefix}-(\\d+)-(\\d+)"[^>]*>`, "g");
  const out = {};
  for (const m of svg.matchAll(re)) {
    const tag = m[0];
    out[`${m[1]}-${m[2]}`] = {
      value: Number(/data-value="(-?[\d.]+)"/.exec(tag)[1]),
      pad: /data-pad="1"/.test(tag),
      chosen: /data-chosen="1"/.test(tag),
    };
  }
  return out;
}
const values = (svg, prefix) => Object.fromEntries(Object.entries(cells(svg, prefix)).map(([k, c]) => [k, c.value]));
const asCells = (m) => Object.fromEntries(m.flatMap((row, r) => row.map((v, c) => [`${r}-${c}`, v])));
const firstCells = (m, n) => Object.fromEntries(Object.entries(asCells(m)).slice(0, n));

function windowAt(svg, name) {
  const m = new RegExp(`data-window="${name}" data-row="(\\d+)" data-col="(\\d+)"`).exec(svg);
  return m ? [Number(m[1]), Number(m[2])] : null;
}

function deepFreeze(m) {
  m.forEach((row) => Object.freeze(row));
  return Object.freeze(m);
}

// ---------- Pure geometry ----------

test("operations reuse the canonical spatial data and register four lessons", () => {
  assert.equal(O.HOUSE, S.HOUSE);
  assert.equal(O.KERNEL, S.KERNELS.vertical);
  assert.deepEqual(O.POOL_INPUT, POOL_INPUT);
  for (const kind of ["stride", "padding", "maxpool", "averagepool"]) {
    assert.equal(S.LESSONS[`cnn-${kind}`], O.LESSONS[kind]);
    assert.ok(O.LESSONS[kind].frameCount > 1);
  }
});

test("outputSize follows floor((n + 2p - k) / s) + 1 and rejects unsupported geometry", () => {
  assert.equal(O.outputSize(5, 3, 1, 0), 3);
  assert.equal(O.outputSize(5, 3, 2, 0), 2);
  assert.equal(O.outputSize(5, 3, 1, 1), 5);
  assert.equal(O.outputSize(4, 2, 2, 0), 2);
  assert.throws(() => O.outputSize(5, 3, 0, 0), /stride/);
  assert.throws(() => O.outputSize(5, 3, 1, -1), /padding/);
  assert.throws(() => O.outputSize(5, 3, 1.5, 0), /stride/);
  assert.throws(() => O.outputSize(2, 3, 1, 0), /larger/);
});

test("windowPositions moves the top-left corner by the stride, row-major", () => {
  const coords = (list) => list.map((p) => [p.row, p.col]);
  assert.deepEqual(coords(O.windowPositions(5, 3, 2)), [[0, 0], [0, 2], [2, 0], [2, 2]]);
  assert.equal(O.windowPositions(5, 3, 1).length, 9);
  assert.equal(O.windowPositions(5, 3, 1, 1).length, 25);
});

test("convolve: stride 1 equals the existing convolution, stride 2 keeps every other position", () => {
  assert.deepEqual(O.convolve(HOUSE, S.KERNELS.vertical), S.convolve2d(HOUSE, S.KERNELS.vertical));
  assert.deepEqual(O.convolve(HOUSE, S.KERNELS.vertical), VALID_MAP);
  assert.deepEqual(O.convolve(HOUSE, S.KERNELS.vertical, { stride: 2 }), [
    [2, 2],
    [3, 3],
  ]);
  // Deep-learning convention: cross-correlation, the kernel is not flipped.
  assert.deepEqual(O.convolve([[1, 2], [3, 4]], [[1, 0], [0, 0]]), [[1]]);
  assert.throws(() => O.convolve([[1, 2]], [[1]]), /square/);
  assert.throws(() => O.convolve(HOUSE, [[1, NaN], [0, 0]]), /square/);
});

test("padZeros adds a zero border and keeps the original pixels untouched", () => {
  const frozen = deepFreeze(HOUSE.map((row) => row.slice()));
  const padded = O.padZeros(frozen, 1);
  assert.equal(padded.length, 7);
  padded.forEach((row) => assert.equal(row.length, 7));
  for (let i = 0; i < 7; i += 1) {
    for (const [r, c] of [[0, i], [6, i], [i, 0], [i, 6]]) assert.equal(padded[r][c], 0);
  }
  for (let r = 0; r < 5; r += 1) assert.deepEqual(padded[r + 1].slice(1, 6), HOUSE[r]);
  const same = O.padZeros(frozen, 0);
  assert.deepEqual(same, HOUSE);
  assert.notEqual(same, frozen, "padding 0 still returns a copy");
  assert.throws(() => O.padZeros(HOUSE, -1), /padding/);
});

test("padding 1 keeps a 5x5 map; its center equals the unpadded 3x3 map", () => {
  const padded = O.convolve(HOUSE, S.KERNELS.vertical, { padding: 1 });
  assert.deepEqual(padded, PADDED_MAP);
  assert.deepEqual(padded.slice(1, 4).map((row) => row.slice(1, 4)), VALID_MAP);
});

test("pool2d: max and average over the SAME 4x4 input, 2x2 window, stride 2", () => {
  assert.deepEqual(O.pool2d(O.POOL_INPUT, { mode: "max" }), MAX_MAP);
  assert.deepEqual(O.pool2d(O.POOL_INPUT), MAX_MAP, "max is the default");
  assert.deepEqual(O.pool2d(O.POOL_INPUT, { mode: "average" }), AVG_MAP);
  assert.deepEqual(O.poolData().max, MAX_MAP);
  assert.deepEqual(O.poolData().average, AVG_MAP);
  assert.equal(O.poolData().input, O.POOL_INPUT);
});

test("pooling handles negatives, ties and fractional averages exactly", () => {
  assert.deepEqual(O.pool2d([[-1, -3], [-2, -5]], { mode: "max" }), [[-1]]);
  assert.deepEqual(O.pool2d([[-1, -3], [-2, -5]], { mode: "average" }), [[-2.75]]);
  const tie = O.poolWindow([[4, 4], [4, 4]], "max");
  assert.equal(tie.value, 4);
  assert.deepEqual(tie.chosen, [0, 1, 2, 3]);
  const spike = O.poolWindow([[1, 3], [2, 9]], "average");
  assert.deepEqual(spike, { values: [1, 3, 2, 9], sum: 15, value: 3.75, chosen: [0, 1, 2, 3] });
  assert.deepEqual(O.poolWindow([[1, 3], [9, 9]], "max").chosen, [2, 3]);
  assert.throws(() => O.pool2d(POOL_INPUT, { mode: "median" }), /mode/);
  assert.throws(() => O.pool2d([[1]], { size: 2 }), /larger/);
});

test("helpers never mutate their inputs", () => {
  const image = deepFreeze(POOL_INPUT.map((row) => row.slice()));
  const kernel = deepFreeze([[1, 0], [0, -1]]);
  assert.doesNotThrow(() => O.convolve(image, kernel, { stride: 2, padding: 1 }));
  assert.doesNotThrow(() => O.pool2d(image, { mode: "average" }));
  assert.doesNotThrow(() => O.poolWindow(image, "max"));
  assert.deepEqual(image, POOL_INPUT);
});

// ---------- Rendered frames ----------

const frames = (kind) => Array.from({ length: O.LESSONS[kind].frameCount }, (_, f) => O.LESSONS[kind].render(f));

test("every frame of every lesson is a 600x440 SVG with a spoken label", () => {
  for (const kind of Object.keys(O.LESSONS)) {
    frames(kind).forEach((out) => {
      assert.match(out.svg, /^<svg[^>]*class="spatial-svg operation-svg"[^>]*viewBox="0 0 600 440"/);
      assert.ok(out.svg.endsWith("</svg>"));
      assert.ok(out.label.length > 20);
    });
  }
});

test("stride frames: same image, stride 1 travels 1 px, stride 2 travels 2 px, maps fill progressively", () => {
  const p1 = O.windowPositions(5, 3, 1);
  const p2 = O.windowPositions(5, 3, 2);
  const all = frames("stride");
  assert.equal(all.length, 10);
  all.slice(0, 9).forEach(({ svg, label }, t) => {
    assert.deepEqual(values(svg, "s1-img"), asCells(HOUSE));
    assert.deepEqual(values(svg, "s2-img"), asCells(HOUSE));
    assert.deepEqual(windowAt(svg, "s1"), [p1[t].row, p1[t].col]);
    assert.deepEqual(values(svg, "s1-out"), firstCells(VALID_MAP, t + 1));
    assert.deepEqual(values(svg, "s2-out"), firstCells([[2, 2], [3, 3]], Math.min(t + 1, 4)));
    if (t < 4) {
      assert.deepEqual(windowAt(svg, "s2"), [p2[t].row, p2[t].col]);
      assert.match(label, new RegExp(`paso 2: posición ${t + 1}/4`));
    } else {
      assert.equal(windowAt(svg, "s2"), null, "stride 2 already finished");
    }
    assert.match(label, new RegExp(`paso 1: posición ${t + 1}/9`));
  });
  const last = all[9];
  assert.deepEqual(values(last.svg, "s1-out"), asCells(VALID_MAP));
  assert.deepEqual(values(last.svg, "s2-out"), asCells([[2, 2], [3, 3]]));
  assert.match(last.label, /3×3/);
  assert.match(last.label, /2×2/);
});

test("padding frames: shrink to 3x3, then a zero border keeps 5x5 without new information", () => {
  const all = frames("padding");
  assert.equal(all.length, 9 + 25 + 1);
  all.slice(0, 9).forEach(({ svg }, t) => {
    const input = cells(svg, "in");
    assert.equal(Object.keys(input).length, 25);
    assert.ok(Object.values(input).every((c) => !c.pad), "no padding yet");
    assert.deepEqual(values(svg, "out"), firstCells(VALID_MAP, t + 1));
    assert.deepEqual(windowAt(svg, "pad"), [Math.floor(t / 3), t % 3]);
  });
  all.slice(9, 34).forEach(({ svg }, t) => {
    const input = cells(svg, "in");
    const padCells = Object.entries(input).filter(([, c]) => c.pad);
    assert.equal(Object.keys(input).length, 49);
    assert.equal(padCells.length, 24);
    padCells.forEach(([k, c]) => {
      const [r, col] = k.split("-").map(Number);
      assert.ok(r === 0 || r === 6 || col === 0 || col === 6, `pad cell ${k} is on the border`);
      assert.equal(c.value, 0);
    });
    for (let r = 0; r < 5; r += 1) {
      for (let c = 0; c < 5; c += 1) assert.equal(input[`${r + 1}-${c + 1}`].value, HOUSE[r][c]);
    }
    assert.deepEqual(values(svg, "out"), firstCells(PADDED_MAP, t + 1));
    assert.deepEqual(windowAt(svg, "pad"), [Math.floor(t / 5), t % 5]);
  });
  // The first padded window covers five border zeros and four real pixels.
  const corner = Object.values(cells(all[9].svg, "patch"));
  assert.equal(corner.filter((c) => c.pad).length, 5);
  assert.match(all[9].label, /0/);
  const last = all[34];
  assert.deepEqual(values(last.svg, "out"), asCells(PADDED_MAP));
  assert.match(last.label, /no agregan información/);
});

test("max pooling frames: highlight four values, then deliver the maximum", () => {
  const all = frames("maxpool");
  assert.equal(all.length, 9);
  for (let w = 0; w < 4; w += 1) {
    const look = all[2 * w];
    const pick = all[2 * w + 1];
    const [r, c] = [Math.floor(w / 2) * 2, (w % 2) * 2];
    [look, pick].forEach(({ svg }) => {
      assert.deepEqual(values(svg, "in"), asCells(POOL_INPUT));
      assert.deepEqual(windowAt(svg, "pool"), [r, c]);
    });
    assert.deepEqual(values(look.svg, "out"), firstCells(MAX_MAP, w));
    assert.deepEqual(values(pick.svg, "out"), firstCells(MAX_MAP, w + 1));
    const chosen = Object.entries(cells(pick.svg, "in")).filter(([, cell]) => cell.chosen);
    assert.ok(chosen.length >= 1);
    chosen.forEach(([, cell]) => assert.equal(cell.value, MAX_MAP[r / 2][c / 2]));
    assert.match(pick.label, new RegExp(`= ${MAX_MAP[r / 2][c / 2]}`));
  }
  assert.equal(Object.values(cells(all[7].svg, "in")).filter((c) => c.chosen).length, 4, "tie: all four 4s");
  assert.deepEqual(values(all[8].svg, "out"), asCells(MAX_MAP));
  assert.match(all[8].svg, /no la garantiza/, "tolerance to small shifts, never guaranteed invariance");
  assert.match(all[8].svg, /sin pesos/);
});

test("average pooling frames: same input, sum/4 delivered, max shown alongside for comparison", () => {
  const all = frames("averagepool");
  assert.equal(all.length, 9);
  const sums = [15, 4, 6, 16];
  all.forEach(({ svg }) => {
    assert.deepEqual(values(svg, "in"), asCells(POOL_INPUT));
    assert.deepEqual(values(svg, "max"), asCells(MAX_MAP), "max reference is in the same frame");
  });
  for (let w = 0; w < 4; w += 1) {
    const avg = AVG_MAP[Math.floor(w / 2)][w % 2];
    assert.deepEqual(values(all[2 * w].svg, "out"), firstCells(AVG_MAP, w));
    assert.deepEqual(values(all[2 * w + 1].svg, "out"), firstCells(AVG_MAP, w + 1));
    assert.match(all[2 * w + 1].label, new RegExp(`${sums[w]} / 4 = ${String(avg).replace(".", "\\.")}`));
  }
  assert.deepEqual(values(all[8].svg, "out"), asCells(AVG_MAP));
  assert.match(all[8].label, /9/);
  assert.match(all[8].label, /3\.75/);
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
  doc.querySelectorAll = (sel) => (sel === "[data-operation-lesson]" ? doc.mounts : []);
  const notify = () => observers.forEach((o) => o.cb([]));
  const lessonOnSlide = (kind, active) => {
    const slide = new El("section");
    slide.classList.add("slide");
    if (active) slide.classList.add("is-active");
    const el = new El("div");
    el.setAttribute("data-operation-lesson", kind);
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

test("autoMount builds each operation lesson and plays only on the active, visible slide", () => {
  const dom = fakeDom();
  const slides = ["stride", "padding", "maxpool", "averagepool"].map((kind) => dom.lessonOnSlide(kind, false));
  dom.lessonOnSlide("unknown", false);
  const insts = O.autoMount(dom.doc);
  assert.equal(insts.length, 4, "unknown kinds are skipped");
  insts.forEach((inst) => {
    assert.ok(inst.root.classList.contains("operation-lesson"));
    assert.match(inst.stage.innerHTML, /^<svg[^>]*viewBox="0 0 600 440"/);
    assert.equal(inst.label.getAttribute("aria-live"), "polite");
    assert.equal(inst.playButton.textContent, "Pausar");
    assert.equal(inst.nextButton.textContent, "Paso siguiente");
  });
  assert.equal(dom.sched.timers.size, 0, "inactive slides: no timer");
  assert.equal(O.autoMount(dom.doc)[0], insts[0], "mount is idempotent");

  slides[2].slide.classList.add("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 1, "only the active slide plays");
  dom.sched.tick();
  assert.equal(insts[2].player.state.frame, 1);
  assert.equal(insts[3].player.state.frame, 0, "state stays isolated per widget");
  dom.doc.hidden = true;
  dom.doc.fire("visibilitychange");
  assert.equal(dom.sched.timers.size, 0, "hidden tab clears the timer");
  dom.doc.hidden = false;
  dom.doc.fire("visibilitychange");
  slides[2].slide.classList.remove("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 0, "leaving the slide clears the timer");
  insts.forEach((inst) => inst.destroy());
  assert.throws(() => O.mount(dom.lessonOnSlide("nope", true).el), /Unknown operation lesson/);
});

test("Play/Pause and Step buttons; deck keys stay scoped to the widget", () => {
  const dom = fakeDom();
  const { el } = dom.lessonOnSlide("padding", true);
  const inst = O.mount(el);
  assert.equal(dom.sched.timers.size, 1);
  inst.playButton.dispatch("click", { detail: 1 });
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 1 });
  assert.equal(inst.player.state.frame, 1);
  const end = key("End", el);
  el.dispatch("keydown", end);
  assert.equal(end.stopped, true);
  assert.equal(inst.player.state.frame, O.LESSONS.padding.frameCount - 1);
  assert.match(inst.label.textContent, /no agregan información/);
  el.dispatch("keydown", key("ArrowRight", el));
  assert.equal(inst.player.state.frame, 0, "wraps around");
  const other = key("f", el);
  el.dispatch("keydown", other);
  assert.equal(other.stopped, false, "non-navigation keys reach the deck");
  inst.destroy();
});

test("reduced motion mounts paused but manual step works", () => {
  const dom = fakeDom({ reduced: true });
  const { el } = dom.lessonOnSlide("stride", true);
  const inst = O.mount(el);
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst.player.state.frame, 1);
  assert.match(inst.label.textContent, /paso 1: posición 2\/9/);
});
