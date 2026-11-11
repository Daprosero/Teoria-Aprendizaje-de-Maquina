"use strict";

// Deterministic checks for the spatial convolution lessons (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const S = require(path.join(__dirname, "..", "docs", "pregrado", "animations", "spatial.js"));

const HOUSE = [
  [0, 0, 1, 0, 0],
  [0, 1, 0, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 1, 1, 0],
];
const VERTICAL = [
  [0, 1, 0],
  [0, 1, 0],
  [0, 1, 0],
];
const HORIZONTAL = [
  [0, 0, 0],
  [1, 1, 1],
  [0, 0, 0],
];

function cellValues(svg, prefix) {
  const re = new RegExp(`data-cell="${prefix}-(\\d+)-(\\d+)"[^>]*data-value="(-?[\\d.]+)"`, "g");
  const out = {};
  for (const m of svg.matchAll(re)) out[`${m[1]}-${m[2]}`] = Number(m[3]);
  return out;
}

// ---------- Pure helpers ----------

test("canonical house and kernels match the feature document", () => {
  assert.deepEqual(S.HOUSE, HOUSE);
  assert.deepEqual(S.KERNELS.vertical, VERTICAL);
  assert.deepEqual(S.KERNELS.horizontal, HORIZONTAL);
});

test("valid stride-1 convolution of the house gives 3x3 maps", () => {
  const v = S.convolve2d(HOUSE, VERTICAL);
  const h = S.convolve2d(HOUSE, HORIZONTAL);
  assert.deepEqual(v, [
    [2, 2, 2],
    [3, 1, 3],
    [3, 2, 3],
  ]);
  assert.deepEqual(h, [
    [1, 2, 1],
    [3, 3, 3],
    [1, 2, 1],
  ]);
});

test("dotTerms exposes every multiplication and the sum", () => {
  const patch = S.windowAt(HOUSE, 1, 0, 3);
  assert.deepEqual(patch, [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
  ]);
  const { terms, sum } = S.dotTerms(patch, VERTICAL);
  assert.equal(terms.length, 9);
  assert.equal(sum, 3);
  assert.deepEqual(terms[1], { x: 1, w: 1, product: 1 });
});

test("padding into a 7x7 canvas shifts the whole house without cropping", () => {
  const a = S.padInto(HOUSE, 7, 1, 0);
  const b = S.padInto(HOUSE, 7, 1, 1);
  const total = (m) => m.flat().reduce((s, x) => s + x, 0);
  assert.equal(a.length, 7);
  assert.equal(a[0].length, 7);
  assert.equal(total(a), total(HOUSE));
  assert.equal(total(b), total(HOUSE));
  for (let r = 0; r < 7; r += 1) {
    for (let c = 0; c < 6; c += 1) assert.equal(b[r][c + 1], a[r][c]);
  }
});

test("the same shared filter produces a map shifted one column", () => {
  const { maps } = S.comparisonData();
  assert.equal(maps[0].length, 5);
  assert.equal(maps[0][0].length, 5);
  for (let r = 0; r < 5; r += 1) {
    for (let c = 0; c < 4; c += 1) assert.equal(maps[1][r][c + 1], maps[0][r][c]);
  }
});

test("parameter counts distinguish per-neuron and whole-layer totals", () => {
  assert.deepEqual(S.denseParams(49, 25), { perNeuron: 50, neurons: 25, total: 1250 });
  assert.deepEqual(S.convParams(3, 1, 1), { perFilter: 10, filters: 1, total: 10 });
});

test("dense neurons have their own independent weights", () => {
  const w0 = S.denseWeights(0, 49);
  const w1 = S.denseWeights(1, 49);
  assert.equal(w0.length, 49);
  assert.notDeepEqual(w0, w1);
  assert.deepEqual(S.denseWeights(0, 49), w0, "deterministic");
});

// ---------- Render frames ----------

test("comparison frames reuse the same 9 conv weights while dense weights change", () => {
  assert.equal(S.LESSONS.comparison.frameCount, 50);
  const f0 = S.LESSONS.comparison.render(0);
  const f1 = S.LESSONS.comparison.render(1);
  const k0 = cellValues(f0.svg, "kernel");
  const k1 = cellValues(f1.svg, "kernel");
  assert.equal(Object.keys(k0).length, 9);
  assert.deepEqual(k0, k1);
  const d0 = cellValues(f0.svg, "dense-w");
  const d1 = cellValues(f1.svg, "dense-w");
  assert.equal(Object.keys(d0).length, 49);
  assert.notDeepEqual(d0, d1);
  for (const f of [f0, f1]) {
    assert.match(f.svg, /49 \+ 1 = 50/);
    assert.match(f.svg, /25 × 50 = 1250/);
    assert.match(f.svg, /9 \+ 1 = 10/);
    assert.match(f.svg, /Por neurona/);
    assert.match(f.svg, /Capa completa/);
  }
  assert.match(f0.label, /Posición 1\/25/);
  assert.match(f0.label, /49 pesos propios/);
  assert.match(f0.label, /mismos 9 pesos/);
});

test("comparison conv output fills progressively with computed values", () => {
  const { maps } = S.comparisonData();
  const f = S.LESSONS.comparison.render(6);
  const out = cellValues(f.svg, "conv-out");
  assert.equal(Object.keys(out).length, 7);
  assert.equal(out["1-1"], maps[0][1][1]);
  const shifted = S.LESSONS.comparison.render(25);
  assert.match(shifted.label, /desplazada 1 píxel/);
  assert.equal(cellValues(shifted.svg, "conv-out")["0-0"], maps[1][0][0]);
});

test("comparison copy avoids overclaiming", () => {
  const all = Array.from({ length: 50 }, (_, i) => S.LESSONS.comparison.render(i));
  for (const f of all) {
    assert.doesNotMatch(f.svg + f.label, /no puede aprender|garantiza/i);
  }
});

test("kernel frames show patch, kernel, products, sum and a growing map", () => {
  const lesson = S.LESSONS.kernel;
  assert.equal(lesson.frameCount, 9);
  const expected = S.convolve2d(HOUSE, VERTICAL);
  for (let i = 0; i < 9; i += 1) {
    const f = lesson.render(i);
    const r = Math.floor(i / 3);
    const c = i % 3;
    assert.equal(Object.keys(cellValues(f.svg, "patch")).length, 9);
    assert.deepEqual(Object.values(cellValues(f.svg, "kernel")), VERTICAL.flat());
    const out = cellValues(f.svg, "out");
    assert.equal(Object.keys(out).length, i + 1);
    assert.equal(out[`${r}-${c}`], expected[r][c]);
    assert.match(f.svg, new RegExp(`= ${expected[r][c]}<`));
    assert.match(f.label, new RegExp(`= ${expected[r][c]}`));
    assert.match(f.svg, /ilustrativo/);
  }
});

test("filter frames build three maps from the same input and flag redundancy", () => {
  const lesson = S.LESSONS.filters;
  assert.equal(lesson.frameCount, 28);
  assert.equal(S.FILTER_BANK.length, 3);
  assert.deepEqual(S.FILTER_BANK[2].kernel, S.FILTER_BANK[0].kernel);
  assert.notEqual(S.FILTER_BANK[2].kernel, S.FILTER_BANK[0].kernel, "computed separately");
  const last = lesson.render(27);
  const maps = [0, 1, 2].map((i) => cellValues(last.svg, `map${i}`));
  maps.forEach((m) => assert.equal(Object.keys(m).length, 9));
  assert.deepEqual(maps[2], maps[0]);
  assert.notDeepEqual(maps[1], maps[0]);
  assert.deepEqual(Object.values(maps[1]), S.convolve2d(HOUSE, HORIZONTAL).flat());
  assert.match(last.svg, /redundancia/i);
  assert.match(last.svg, /según la tarea/);
  assert.match(last.label, /misma casa/);
  const mid = lesson.render(10);
  assert.equal(Object.keys(cellValues(mid.svg, "map1")).length, 2);
  assert.equal(Object.keys(cellValues(mid.svg, "map2")).length, 0);
});

// ---------- Player ----------

function fakeScheduler() {
  const timers = new Map();
  let id = 0;
  return {
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
}

test("player runs only while active and stops timers when hidden", () => {
  const sched = fakeScheduler();
  let active = false;
  const frames = [];
  const p = S.createPlayer({
    frameCount: 4,
    onFrame: (f) => frames.push(f),
    isActive: () => active,
    reducedMotion: false,
    scheduler: sched,
  });
  assert.equal(sched.timers.size, 0);
  active = true;
  p.sync();
  assert.equal(sched.timers.size, 1);
  sched.tick();
  assert.equal(p.state.frame, 1);
  active = false;
  p.sync();
  assert.equal(sched.timers.size, 0);
  p.destroy();
});

test("pause, manual step and reduced motion", () => {
  const sched = fakeScheduler();
  const p = S.createPlayer({ frameCount: 3, onFrame() {}, isActive: () => true, reducedMotion: false, scheduler: sched });
  assert.equal(p.state.playing, true);
  p.pause();
  assert.equal(sched.timers.size, 0);
  p.next();
  p.next();
  p.next();
  assert.equal(p.state.frame, 0, "wraps around");
  p.play();
  assert.equal(sched.timers.size, 1);
  p.next();
  assert.equal(p.state.playing, false, "manual step pauses");
  assert.equal(sched.timers.size, 0);

  const s2 = fakeScheduler();
  const r = S.createPlayer({ frameCount: 3, onFrame() {}, isActive: () => true, reducedMotion: true, scheduler: s2 });
  assert.equal(r.state.playing, false);
  assert.equal(s2.timers.size, 0);
  r.next();
  assert.equal(r.state.frame, 1);
});

// ---------- Mount with a minimal fake DOM ----------

function fakeDom({ reduced = false } = {}) {
  const sched = fakeScheduler();
  const observers = [];
  class ClassList {
    constructor() {
      this.set = new Set();
    }
    add(...c) {
      c.forEach((x) => this.set.add(x));
    }
    remove(...c) {
      c.forEach((x) => this.set.delete(x));
    }
    contains(c) {
      return this.set.has(c);
    }
  }
  const doc = { readyState: "complete", mounts: [] };
  class El {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.children = [];
      this.attrs = {};
      this.listeners = {};
      this.classList = new ClassList();
      this.textContent = "";
      this.innerHTML = "";
      this.parentNode = null;
      this.ownerDocument = doc;
    }
    appendChild(c) {
      c.parentNode = this;
      this.children.push(c);
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
      while (n) {
        if (n.classList.contains(sel.slice(1))) return n;
        n = n.parentNode;
      }
      return null;
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
  const win = {
    setInterval: sched.setInterval,
    clearInterval: sched.clearInterval,
    matchMedia: () => ({ matches: reduced, addEventListener() {} }),
    MutationObserver: MO,
  };
  doc.defaultView = win;
  doc.createElement = (t) => new El(t);
  doc.querySelectorAll = () => doc.mounts;
  const notify = () => observers.forEach((o) => o.cb([]));
  return { doc, win, sched, El, notify };
}

function keyEvent(key, target) {
  return {
    key,
    target,
    stopped: false,
    prevented: false,
    stopPropagation() {
      this.stopped = true;
    },
    preventDefault() {
      this.prevented = true;
    },
  };
}

test("autoMount builds each lesson with SVG, live label and controls", () => {
  const dom = fakeDom();
  const slide = new dom.El("section");
  slide.classList.add("slide");
  for (const kind of ["comparison", "kernel", "filters"]) {
    const el = new dom.El("div");
    el.setAttribute("data-spatial-lesson", kind);
    slide.appendChild(el);
    dom.doc.mounts.push(el);
  }
  const instances = S.autoMount(dom.doc);
  assert.equal(instances.length, 3);
  for (const inst of instances) {
    assert.match(inst.stage.innerHTML, /^<svg[^>]*viewBox="0 0 600 440"/);
    assert.equal(inst.label.getAttribute("aria-live"), "polite");
    assert.ok(inst.label.textContent.length > 0);
    assert.equal(inst.playButton.textContent, "Pausar");
    assert.equal(inst.nextButton.textContent, "Paso siguiente");
  }
  assert.equal(dom.sched.timers.size, 0, "inactive slide: no timers");
  assert.equal(S.autoMount(dom.doc)[0], instances[0], "mount is idempotent");

  slide.classList.add("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 3);
  const before = instances[1].label.textContent;
  dom.sched.tick();
  assert.notEqual(instances[1].label.textContent, before);

  slide.classList.remove("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 0, "hidden slide clears timers");
});

test("buttons pause and step; deck navigation keys stay inside the widget", () => {
  const dom = fakeDom();
  const slide = new dom.El("section");
  slide.classList.add("slide", "is-active");
  const el = new dom.El("div");
  slide.appendChild(el);
  const inst = S.mount(el, "kernel");
  assert.equal(dom.sched.timers.size, 1);

  inst.playButton.dispatch("click", { detail: 1 });
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");

  inst.nextButton.dispatch("click", { detail: 1 });
  assert.equal(inst.player.state.frame, 1);
  assert.match(inst.label.textContent, /Posición 2\/9/);

  for (const key of [" ", "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End"]) {
    const ev = keyEvent(key, el);
    el.dispatch("keydown", ev);
    assert.equal(ev.stopped, true, key);
  }
  const ev = keyEvent("End", el);
  el.dispatch("keydown", ev);
  assert.equal(inst.player.state.frame, 8);
  const home = keyEvent("Home", el);
  el.dispatch("keydown", home);
  assert.equal(inst.player.state.frame, 0);
  const other = keyEvent("f", el);
  el.dispatch("keydown", other);
  assert.equal(other.stopped, false, "fullscreen key still reaches the deck");
  inst.destroy();
});

test("reduced motion mounts paused but manual step works", () => {
  const dom = fakeDom({ reduced: true });
  const slide = new dom.El("section");
  slide.classList.add("slide", "is-active");
  const el = new dom.El("div");
  slide.appendChild(el);
  const inst = S.mount(el, "filters");
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst.player.state.frame, 1);
});
