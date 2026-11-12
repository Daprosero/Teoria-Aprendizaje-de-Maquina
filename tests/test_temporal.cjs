"use strict";

// Deterministic checks for the time-signal, autoregression and train/test split lessons (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const dir = path.join(__dirname, "..", "docs", "pregrado", "animations");
const S = require(path.join(dir, "spatial.js"));
const T = require(path.join(dir, "temporal.js"));

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const round2 = (v) => Math.round(v * 100) / 100;

// Every <circle> that carries data-t, with all of its data-* attributes.
function points(svg, plot) {
  const out = [];
  for (const m of svg.matchAll(/<circle\b[^>]*>/g)) {
    const attrs = {};
    for (const a of m[0].matchAll(/data-([\w-]+)="([^"]*)"/g)) attrs[a[1]] = a[2];
    if (!("t" in attrs)) continue;
    if (plot && attrs.plot !== plot) continue;
    out.push({ ...attrs, t: Number(attrs.t), y: Number(attrs.y) });
  }
  return out;
}
const times = (pts, set) => pts.filter((p) => !set || p.set === set).map((p) => p.t);
const count = (svg, name) => Number(new RegExp(`data-count="${name}" data-value="(\\d+)"`).exec(svg)[1]);

// ---------- Canonical data and pure helpers ----------

test("temporal registers three namespaced lessons on the shared spatial registry", () => {
  for (const kind of ["signal", "autoregression", "split"]) {
    assert.equal(S.LESSONS[`temporal-${kind}`], T.LESSONS[kind]);
    assert.ok(T.LESSONS[kind].frameCount > 1);
  }
  assert.equal(S.LESSONS.kernel.frameCount, 9, "existing lessons untouched");
});

test("SIGNAL: 100 frozen points, t = 0..99, trend + 20-step cycle + bounded seeded noise", () => {
  assert.equal(T.SIGNAL.length, 100);
  assert.ok(Object.isFrozen(T.SIGNAL) && Object.isFrozen(T.SIGNAL[0]));
  assert.deepEqual(T.SIGNAL.map((p) => p.t), range(0, 100));
  T.SIGNAL.forEach((p) => {
    assert.ok(Number.isFinite(p.y));
    assert.equal(p.y, round2(p.y), "two decimals, exactly what the figures print");
    const clean = T.trendAt(p.t) + T.seasonalAt(p.t);
    assert.ok(Math.abs(p.y - clean) <= T.NOISE_AMPLITUDE + 0.005, `t=${p.t}`);
  });
  assert.equal(T.seasonalAt(5), T.seasonalAt(25), "the cycle repeats every 20 steps");
  assert.deepEqual(T.signalValues(), T.SIGNAL.map((p) => p.y));
  assert.deepEqual(T.generateSignal(), T.SIGNAL, "same seed, same values");
  assert.notDeepEqual(T.generateSignal(8), T.SIGNAL, "another seed changes only the noise");
});

test("mulberry32 and seededShuffle are deterministic and never mutate their input", () => {
  const a = T.mulberry32(7);
  const b = T.mulberry32(7);
  const seq = range(0, 50).map(() => a());
  assert.deepEqual(seq, range(0, 50).map(() => b()));
  assert.ok(seq.every((v) => v >= 0 && v < 1));
  assert.notDeepEqual(seq.slice(0, 5), range(0, 5).map(T.mulberry32(8)));
  assert.throws(() => T.mulberry32(-1), /seed/);
  assert.throws(() => T.mulberry32(1.5), /seed/);

  const input = Object.freeze(range(0, 20));
  const shuffled = T.seededShuffle(input, 3);
  assert.deepEqual(input, range(0, 20));
  assert.deepEqual([...shuffled].sort((x, y) => x - y), range(0, 20), "a permutation");
  assert.notDeepEqual(shuffled, range(0, 20));
  assert.deepEqual(T.seededShuffle(input, 3), shuffled);
});

test("partition random: exact counts, disjoint, covers 0..99, sorted, reproducible per seed", () => {
  for (const fraction of [0.1, 0.2, 0.3, 0.4]) {
    const p = T.partition({ mode: "random", testFraction: fraction, seed: 7 });
    const expected = Math.round(100 * fraction);
    assert.equal(p.test.length, expected);
    assert.equal(p.train.length, 100 - expected);
    assert.deepEqual([...p.train, ...p.test].sort((x, y) => x - y), range(0, 100));
    assert.equal(new Set([...p.train, ...p.test]).size, 100, "disjoint");
    assert.deepEqual(p.train, [...p.train].sort((x, y) => x - y));
    assert.deepEqual(p.test, [...p.test].sort((x, y) => x - y));
    assert.deepEqual(T.partition({ mode: "random", testFraction: fraction, seed: 7 }), p);
  }
  const s7 = T.partition({ mode: "random", seed: 7 });
  const s8 = T.partition({ mode: "random", seed: 8 });
  assert.notDeepEqual(s7.test, s8.test, "another seed draws another test set");
  assert.ok(Math.min(...s7.test) < 50 && Math.max(...s7.test) >= 50, "random test spreads over the axis");
  assert.deepEqual(T.partition(), s7, "defaults: random, 20 %, seed 7");
});

test("partition chronological: the test set is the last contiguous block, the seed is irrelevant", () => {
  const p = T.partition({ mode: "chronological" });
  assert.deepEqual(p.train, range(0, 80));
  assert.deepEqual(p.test, range(80, 100));
  assert.ok(Math.max(...p.train) < Math.min(...p.test));
  const p30 = T.partition({ mode: "chronological", testFraction: 0.3, seed: 99 });
  assert.deepEqual(p30.test, range(70, 100));
  assert.deepEqual(T.partition({ mode: "chronological", testFraction: 0.3, seed: 1 }).test, p30.test);
  assert.equal(p30.seed, null);
});

test("partition rejects unknown modes, fractions and seeds", () => {
  assert.throws(() => T.partition({ mode: "kfold" }), /mode/);
  assert.throws(() => T.partition({ mode: "random", testFraction: 0 }), /testFraction/);
  assert.throws(() => T.partition({ mode: "random", testFraction: 1 }), /testFraction/);
  assert.throws(() => T.partition({ mode: "random", testFraction: 0.001 }), /testFraction/);
  assert.throws(() => T.partition({ mode: "random", seed: -3 }), /seed/);
});

test("arForecast uses only the three previous values, newest lag first", () => {
  const values = Object.freeze([1, 2, 3, 4, 5]);
  const f = T.arForecast(values, 3);
  assert.deepEqual(T.AR_COEFFS, [0.6, 0.25, 0.1]);
  assert.deepEqual(f.lags.map((l) => l.t), [2, 1, 0]);
  assert.deepEqual(f.lags.map((l) => l.y), [3, 2, 1]);
  assert.deepEqual(f.lags.map((l) => l.weight), [0.6, 0.25, 0.1]);
  assert.ok(Math.abs(f.prediction - (0.6 * 3 + 0.25 * 2 + 0.1 * 1)) < 1e-12);
  assert.equal(f.observed, 4);
  assert.ok(Math.abs(f.error - (4 - f.prediction)) < 1e-12);
  assert.deepEqual(values, [1, 2, 3, 4, 5], "input not mutated");

  // Changing the target or anything after it never changes the forecast.
  const leaked = [1, 2, 3, 400, 500];
  assert.equal(T.arForecast(leaked, 3).prediction, f.prediction);

  assert.throws(() => T.arForecast(values, 2), /target/);
  assert.throws(() => T.arForecast(values, 5), /target/);
  assert.throws(() => T.arForecast(values, 3.5), /target/);
  assert.throws(() => T.arForecast([1, NaN, 3, 4], 3), /values/);

  const y = T.signalValues();
  const s = T.arForecast(y, 31);
  assert.equal(s.prediction, 0.6 * y[30] + 0.25 * y[29] + 0.1 * y[28]);
  // The worked example printed in temporal-notes.md.
  assert.deepEqual(y.slice(28, 32), [11.78, 12.27, 11.26, 10.77]);
  assert.equal(s.prediction.toFixed(2), "11.00");
});

test("lagTable rows keep features strictly before their target", () => {
  const rows = T.lagTable(T.signalValues());
  assert.equal(rows.length, 97);
  assert.equal(rows[0].target, 3);
  rows.forEach((row) => {
    assert.deepEqual(row.featureTimes, [row.target - 1, row.target - 2, row.target - 3]);
    assert.ok(Math.max(...row.featureTimes) < row.target);
    assert.deepEqual(row.features, row.featureTimes.map((t) => T.SIGNAL[t].y));
    assert.equal(row.y, T.SIGNAL[row.target].y);
  });
});

test("contiguousRuns splits indices wherever an observation is missing", () => {
  assert.deepEqual(T.contiguousRuns([0, 1, 2, 5, 6, 9]), [[0, 1, 2], [5, 6], [9]]);
  assert.deepEqual(T.contiguousRuns([]), []);
});

// ---------- Rendered frames ----------

test("signal lesson: grid intro, then points appear in time order with their real values", () => {
  const lesson = T.LESSONS.signal;
  const intro = lesson.render(0);
  assert.match(intro.svg, /^<svg[^>]*viewBox="0 0 600 440"/);
  assert.equal(points(intro.svg).length, 0);
  assert.match(intro.svg, /data-role="image-grid"/);
  assert.match(lesson.render(1).svg, /data-role="time-axis"/);

  for (let k = 0; k < 20; k += 1) {
    const out = lesson.render(2 + k);
    const pts = points(out.svg, "signal");
    const end = 5 * k + 4;
    assert.deepEqual(times(pts), range(0, end + 1));
    pts.forEach((p) => assert.equal(p.y, T.SIGNAL[p.t].y));
    assert.match(out.svg, new RegExp(`data-cursor="${end}"`));
    assert.ok(out.label.includes(`t = ${end}`) && out.label.includes(T.SIGNAL[end].y.toFixed(2)), out.label);
  }
  const final = lesson.render(lesson.frameCount - 1);
  assert.deepEqual(times(points(final.svg, "signal")), range(0, 100));
  assert.match(final.svg, /data-role="trend"/);
  assert.match(final.svg, /sintética/);
});

test("autoregression lesson: future hidden before the forecast, numbers from arForecast, truth revealed last", () => {
  const lesson = T.LESSONS.autoregression;
  const y = T.signalValues();
  assert.equal(lesson.frameCount, T.AR_TARGETS.length * 3);
  T.AR_TARGETS.forEach((target, i) => {
    const f = T.arForecast(y, target);
    const [origin, forecast, reveal] = [0, 1, 2].map((s) => lesson.render(3 * i + s).svg);
    const pred = f.prediction.toFixed(2);

    for (const svg of [origin, forecast]) {
      const observed = points(svg).filter((p) => p.set === "observed");
      assert.ok(observed.every((p) => p.t < target), `no future truth before reveal, target ${target}`);
      assert.equal(Math.max(...times(observed)), target - 1);
      observed.forEach((p) => assert.equal(p.y, T.SIGNAL[p.t].y));
      const lags = points(svg).filter((p) => p.set === "lag");
      assert.deepEqual(times(lags), [target - 1, target - 2, target - 3]);
      lags.forEach((p) => assert.equal(p.y, T.SIGNAL[p.t].y));
      assert.doesNotMatch(svg, /data-set="target-observed"/);
    }
    assert.doesNotMatch(origin, /data-prediction=/, "no number before the forecast step");
    assert.match(forecast, new RegExp(`data-prediction="${pred}"`));
    assert.ok(forecast.includes(`= ${pred}`));
    for (const lag of f.lags) assert.ok(forecast.includes(`${lag.weight}·${lag.y.toFixed(2)}`));

    const truth = points(reveal).filter((p) => p.set === "target-observed");
    assert.deepEqual(times(truth), [target]);
    assert.equal(truth[0].y, T.SIGNAL[target].y);
    assert.match(reveal, new RegExp(`data-prediction="${pred}"`));
    assert.ok(reveal.includes(f.error.toFixed(2)));
    assert.ok(points(reveal).every((p) => p.t <= target), "never beyond the target");
  });
  assert.match(lesson.render(0).svg, /no entrenados/);
});

function splitCheck(svg, part) {
  const main = points(svg, "main");
  assert.deepEqual(times(main, "train"), part.train);
  assert.deepEqual(times(main, "test"), part.test);
  main.forEach((p) => assert.equal(p.y, T.SIGNAL[p.t].y, "original time axis and values"));
  assert.deepEqual(times(points(svg, "train-only")), part.train);
  assert.deepEqual(times(points(svg, "test-only")), part.test);
  assert.equal(count(svg, "train"), part.train.length);
  assert.equal(count(svg, "test"), part.test.length);
}

test("split lesson: plotted train/test points are exactly the partition, on the original axis", () => {
  const configs = [
    { mode: "random", testFraction: 0.2, seed: 7 },
    { mode: "random", testFraction: 0.4, seed: 11 },
    { mode: "chronological", testFraction: 0.3, seed: 7 },
  ];
  for (const config of configs) {
    const final = T.renderSplit(T.LESSONS.split.frameCount - 1, config);
    splitCheck(final.svg, T.partition(config));
    assert.match(final.svg, /data-role="reference"/);
    assert.doesNotMatch(final.svg, /<path[^>]*data-plot="(train|test)-only"/, "no lines across missing points");
    assert.ok(final.label.includes(`train ${T.partition(config).train.length}`));
  }
  const chrono = T.renderSplit(11, configs[2]).svg;
  assert.match(chrono, /data-role="test-band" data-from="70" data-to="99"/);
  const random = T.renderSplit(11, configs[0]).svg;
  assert.doesNotMatch(random, /data-role="test-band"/);
  assert.match(random, /IID/);
  assert.match(random, /no prueba/);

  // The animation reveals the same assignment from left to right.
  const first = T.renderSplit(0, configs[0]).svg;
  assert.equal(points(first, "main").filter((p) => p.set === "pending").length, 100);
  const mid = T.renderSplit(5, configs[0]).svg;
  const part = T.partition(configs[0]);
  assert.deepEqual(times(points(mid, "main"), "test"), part.test.filter((t) => t < 50));
  assert.deepEqual(times(points(mid, "main"), "pending"), range(50, 100));
  assert.equal(T.LESSONS.split.render(11).svg, T.renderSplit(11, T.DEFAULT_SPLIT).svg);
});

// ---------- Mount with a minimal fake DOM ----------

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
      this.value = "";
      this.disabled = false;
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
      (this.listeners[t] || []).forEach((fn) => fn(ev || {}));
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
  doc.querySelectorAll = (sel) => doc.mounts.filter((el) => el.getAttribute(sel.slice(1, -1)) !== null);
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

function mountOn(dom, kind, active) {
  const slide = new dom.El("section");
  slide.classList.add("slide");
  if (active) slide.classList.add("is-active");
  const el = new dom.El("div");
  el.setAttribute("data-temporal-lesson", kind);
  slide.appendChild(el);
  dom.doc.mounts.push(el);
  return { slide, el };
}

test("autoMount builds the three lessons, runs only on the active slide and is idempotent", () => {
  const dom = fakeDom();
  const slides = ["signal", "autoregression", "split"].map((kind) => mountOn(dom, kind, false));
  const other = new dom.El("div");
  other.setAttribute("data-temporal-lesson", "unknown");
  dom.doc.mounts.push(other);
  const instances = T.autoMount(dom.doc);
  assert.equal(instances.length, 3);
  instances.forEach((inst, i) => {
    assert.equal(inst.kind, ["signal", "autoregression", "split"][i]);
    assert.ok(inst.root.classList.contains("temporal-lesson"));
    assert.match(inst.stage.innerHTML, /^<svg[^>]*viewBox="0 0 600 440"/);
    assert.equal(inst.label.getAttribute("aria-live"), "polite");
    assert.equal(inst.playButton.textContent, "Pausar");
    assert.equal(inst.nextButton.textContent, "Paso siguiente");
  });
  assert.equal(dom.sched.timers.size, 0, "inactive slides: no timers");
  assert.equal(T.autoMount(dom.doc)[2], instances[2], "mount is idempotent");
  assert.equal(slides[2].el.children.filter((c) => c.classList.contains("temporal-options")).length, 1);

  slides.forEach((s) => s.slide.classList.add("is-active"));
  dom.notify();
  assert.equal(dom.sched.timers.size, 3);
  const before = instances[0].label.textContent;
  dom.sched.tick();
  assert.notEqual(instances[0].label.textContent, before);
  slides.forEach((s) => s.slide.classList.remove("is-active"));
  dom.notify();
  assert.equal(dom.sched.timers.size, 0, "hidden slides clear timers");
  assert.throws(() => T.mount(new dom.El("div"), "rnn"), /Unknown temporal lesson/);
});

test("split controls change the plotted partition, the counts and the label", () => {
  const dom = fakeDom();
  const { el } = mountOn(dom, "split", true);
  const inst = T.mount(el);
  const svg = () => inst.stage.innerHTML;
  assert.deepEqual(inst.getConfig(), T.DEFAULT_SPLIT);
  assert.equal(inst.fractionInput.getAttribute("type"), "range");
  assert.deepEqual(["min", "max", "step"].map((k) => inst.fractionInput.getAttribute(k)), ["10", "40", "10"]);
  assert.equal(inst.fractionInput.value, "20");
  assert.equal(inst.modeButtons.random.getAttribute("aria-pressed"), "true");

  // Fraction slider: 30 % random.
  inst.fractionInput.value = "30";
  inst.fractionInput.dispatch("input");
  const p30 = T.partition({ mode: "random", testFraction: 0.3, seed: 7 });
  splitCheck(svg(), p30);
  assert.equal(inst.player.state.frame, T.LESSONS.split.frameCount - 1, "shows the full assignment");
  assert.equal(inst.player.state.playing, false, "a manual change pauses like a manual step");
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.fractionValue.textContent, "30 %");
  assert.ok(inst.label.textContent.includes("test 30"), inst.label.textContent);

  // New seed: another random draw with the same counts.
  inst.seedButton.dispatch("click", { detail: 1 });
  assert.equal(inst.getConfig().seed, 8);
  const p8 = T.partition({ mode: "random", testFraction: 0.3, seed: 8 });
  assert.notDeepEqual(p8.test, p30.test);
  splitCheck(svg(), p8);

  // Chronological: the last 30 % become one test block; the seed button is disabled.
  inst.modeButtons.chronological.dispatch("click", { detail: 1 });
  splitCheck(svg(), T.partition({ mode: "chronological", testFraction: 0.3 }));
  assert.deepEqual(times(points(svg(), "main"), "test"), range(70, 100));
  assert.match(svg(), /data-role="test-band" data-from="70" data-to="99"/);
  assert.equal(inst.modeButtons.chronological.getAttribute("aria-pressed"), "true");
  assert.equal(inst.modeButtons.random.getAttribute("aria-pressed"), "false");
  assert.equal(inst.seedButton.disabled, true);

  inst.fractionInput.value = "10";
  inst.fractionInput.dispatch("input");
  assert.deepEqual(times(points(svg(), "main"), "test"), range(90, 100));
  assert.equal(count(svg(), "train"), 90);

  // Back to random: the seed 8 draw at 10 %.
  inst.modeButtons.random.dispatch("click", { detail: 1 });
  splitCheck(svg(), T.partition({ mode: "random", testFraction: 0.1, seed: 8 }));
  assert.equal(inst.seedButton.disabled, false);

  assert.throws(() => inst.setConfig({ testFraction: 2 }), /testFraction/);
  splitCheck(svg(), T.partition({ mode: "random", testFraction: 0.1, seed: 8 }));

  // Separate instances keep separate state.
  const other = T.mount(mountOn(dom, "split", true).el);
  assert.deepEqual(other.getConfig(), T.DEFAULT_SPLIT);
  splitCheck(T.renderSplit(11, other.getConfig()).svg, T.partition());
});

test("keyboard: deck keys stay in the widget, the slider keeps its own arrow keys", () => {
  const dom = fakeDom();
  const { el } = mountOn(dom, "split", true);
  const inst = T.mount(el);
  inst.player.pause();
  for (const key of [" ", "ArrowRight", "ArrowLeft", "Home", "End"]) {
    const ev = keyEvent(key, el);
    el.dispatch("keydown", ev);
    assert.equal(ev.stopped, true, key);
  }
  const end = keyEvent("End", el);
  el.dispatch("keydown", end);
  assert.equal(inst.player.state.frame, T.LESSONS.split.frameCount - 1);
  const onSlider = keyEvent("ArrowLeft", inst.fractionInput);
  inst.fractionInput.dispatch("keydown", onSlider);
  assert.equal(onSlider.stopped, true, "does not reach the widget stepper or the deck");
  assert.equal(onSlider.prevented, false, "the browser still moves the slider");
  const other = keyEvent("f", el);
  el.dispatch("keydown", other);
  assert.equal(other.stopped, false, "fullscreen key still reaches the deck");
});

test("reduced motion mounts paused; manual step still works", () => {
  const dom = fakeDom({ reduced: true });
  const inst = T.mount(mountOn(dom, "autoregression", true).el);
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst.player.state.frame, 1);
  assert.match(inst.stage.innerHTML, /data-prediction=/);
});
