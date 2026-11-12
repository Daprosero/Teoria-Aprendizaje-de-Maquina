"use strict";

// Deterministic checks for the ordinary-fold, leakage, expanding and sliding validation lessons (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const dir = path.join(__dirname, "..", "docs", "pregrado", "animations");
const S = require(path.join(dir, "spatial.js"));
const T = require(path.join(dir, "temporal.js"));
const V = require(path.join(dir, "validation.js"));

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const sorted = (xs) => [...xs].sort((a, b) => a - b);
const union = (...sets) => sorted(new Set(sets.flat()));
const HOLDOUT = T.partition({ mode: "random", testFraction: 0.2, seed: 7 }).test;
const DEV = T.partition({ mode: "random", testFraction: 0.2, seed: 7 }).train;
const FINAL_BLOCK = range(80, 100);

function attrsOf(tag) {
  const attrs = {};
  for (const a of tag.matchAll(/data-([\w-]+)="([^"]*)"/g)) attrs[a[1]] = a[2];
  return attrs;
}
// Circles that carry data-t (signal points), optionally for one plot.
function points(svg, plot) {
  return [...svg.matchAll(/<circle\b[^>]*>/g)]
    .map((m) => attrsOf(m[0]))
    .filter((a) => "t" in a && (!plot || a.plot === plot))
    .map((a) => ({ ...a, t: Number(a.t), y: Number(a.y) }));
}
// Fold-row cells: <rect data-row=... data-fold=... data-t=... data-role=...>.
function cells(svg, row) {
  return [...svg.matchAll(/<rect\b[^>]*data-row="[^"]*"[^>]*>/g)]
    .map((m) => attrsOf(m[0]))
    .filter((a) => !row || a.row === row)
    .map((a) => ({ ...a, t: Number(a.t), fold: Number(a.fold) }));
}
const times = (items, role, key = "set") => items.filter((p) => p[key] === role).map((p) => p.t);
const count = (svg, name) => Number(new RegExp(`data-count="${name}" data-value="(\\d+)"`).exec(svg)[1]);

function assertDisjointFold(fold) {
  const all = [...fold.train, ...fold.validation, ...fold.test];
  assert.equal(new Set(all).size, all.length, `fold ${fold.index}: train/validation/test are disjoint`);
}

// Every point of the main plot sits on the canonical signal, with its original t.
function assertCanonical(pts) {
  pts.forEach((p) => assert.equal(p.y, T.SIGNAL[p.t].y, `t=${p.t}`));
}

// ---------- Registry ----------

test("validation registers four namespaced lessons on the shared spatial registry", () => {
  for (const kind of ["folds", "leakage", "expanding", "sliding"]) {
    assert.equal(S.LESSONS[`validation-${kind}`], V.LESSONS[kind]);
    assert.ok(V.LESSONS[kind].frameCount > 1);
  }
  assert.equal(S.LESSONS.kernel.frameCount, 9, "existing spatial lessons untouched");
  assert.equal(S.LESSONS["temporal-split"], T.LESSONS.split, "existing temporal lessons untouched");
});

// ---------- Ordinary K-fold with an external holdout ----------

test("iidFolds: one fixed random holdout of 20 is outside all five folds of 64 train / 16 validation", () => {
  const data = V.iidFolds();
  assert.deepEqual(data.holdout, HOLDOUT, "holdout = partition({ mode: random, testFraction: 0.2, seed: 7 }).test");
  assert.deepEqual(data.development, DEV);
  assert.equal(data.folds.length, 5);
  data.folds.forEach((fold, i) => {
    assert.equal(fold.index, i);
    assert.deepEqual(fold.test, HOLDOUT, "exactly the same holdout in every fold");
    assert.equal(fold.train.length, 64);
    assert.equal(fold.validation.length, 16);
    assert.equal(fold.test.length, 20);
    assertDisjointFold(fold);
    assert.deepEqual(union(fold.train, fold.validation, fold.test), range(0, 100));
    assert.deepEqual(fold.train, DEV.filter((t) => !fold.validation.includes(t)), "train = development minus validation");
    assert.deepEqual(fold.train, sorted(fold.train));
    assert.deepEqual(fold.validation, sorted(fold.validation));
    HOLDOUT.forEach((t) => assert.ok(!fold.train.includes(t) && !fold.validation.includes(t), "holdout never used"));
  });
  const validated = data.folds.flatMap((f) => f.validation);
  assert.equal(validated.length, 80);
  assert.deepEqual(sorted(validated), DEV, "each development sample is validated exactly once");
  assert.ok(Math.min(...data.folds[0].validation) < 40 && Math.max(...data.folds[0].validation) > 60, "IID groups spread over time");
});

test("iidFolds is reproducible, returns fresh copies and validates its parameters", () => {
  const a = V.iidFolds();
  a.folds[0].train.push(999);
  a.holdout.length = 0;
  a.development.reverse();
  const b = V.iidFolds();
  assert.deepEqual(b.holdout, HOLDOUT);
  assert.deepEqual(b.development, DEV);
  assert.equal(b.folds[0].train.length, 64);
  assert.notEqual(b.folds[0].test, b.folds[1].test, "each fold has its own copy");
  const options = Object.freeze({ k: 4, seed: 3 });
  const four = V.iidFolds(options);
  assert.deepEqual(options, { k: 4, seed: 3 });
  assert.equal(four.folds.length, 4);
  four.folds.forEach((f) => assert.equal(f.validation.length, 20));
  assert.deepEqual(four.holdout, HOLDOUT, "the fold seed never moves the holdout");
  assert.notDeepEqual(V.iidFolds({ seed: 8 }).folds[0].validation, V.iidFolds().folds[0].validation);
  assert.throws(() => V.iidFolds({ k: 1 }), /k must/);
  assert.throws(() => V.iidFolds({ k: 2.5 }), /k must/);
  assert.throws(() => V.iidFolds({ k: 81 }), /k must/);
  assert.throws(() => V.iidFolds({ seed: -1 }), /seed/);
  assert.equal(T.SIGNAL.length, 100);
});

// ---------- Temporal folds ----------

const EXPANDING = [
  [range(0, 30), range(30, 40)],
  [range(0, 40), range(40, 50)],
  [range(0, 50), range(50, 60)],
  [range(0, 60), range(60, 70)],
  [range(0, 70), range(70, 80)],
];
const SLIDING = [
  [range(0, 30), range(30, 40)],
  [range(10, 40), range(40, 50)],
  [range(20, 50), range(50, 60)],
  [range(30, 60), range(60, 70)],
  [range(40, 70), range(70, 80)],
];

function assertTemporal(data) {
  assert.deepEqual(data.test, FINAL_BLOCK);
  assert.deepEqual(data.test, T.partition({ mode: "chronological", testFraction: 0.2 }).test);
  data.folds.forEach((fold) => {
    assertDisjointFold(fold);
    assert.deepEqual(fold.test, FINAL_BLOCK, "the final test block is identical in every fold");
    assert.ok(Math.max(...fold.train) < Math.min(...fold.validation), "strictly past train before validation");
    assert.ok([...fold.train, ...fold.validation].every((t) => t < 80), "train and validation never reach the test");
    assert.equal(fold.validation.length, 10, "same forecast horizon");
    assert.deepEqual(T.contiguousRuns(fold.train).length, 1, "contiguous train");
    assert.deepEqual(T.contiguousRuns(fold.validation).length, 1, "contiguous validation");
    assert.equal(fold.validation[0], fold.train[fold.train.length - 1] + 1, "validation right after train");
    assert.equal(fold.origin, fold.validation[0]);
    assert.deepEqual(union(fold.discarded, fold.train, fold.validation, fold.unused, fold.test), range(0, 100));
    assert.equal(fold.discarded.length + fold.train.length + fold.validation.length + fold.unused.length + 20, 100);
    assert.deepEqual(fold.unused, range(fold.validation[9] + 1, 80), "unused future development stays outside");
  });
  for (let i = 1; i < data.folds.length; i += 1) {
    assert.equal(data.folds[i].validation[0], data.folds[i - 1].validation[0] + 10, "validation moves forward by 10");
  }
}

test("temporalFolds expanding: train 0..29 → 0..69, grows by 10, always a strict superset", () => {
  const data = V.temporalFolds({ kind: "expanding" });
  assert.equal(data.kind, "expanding");
  assert.deepEqual(data.folds.map((f) => [f.train, f.validation]), EXPANDING);
  assertTemporal(data);
  data.folds.forEach((f) => assert.deepEqual(f.discarded, [], "expanding keeps all the past"));
  for (let i = 1; i < 5; i += 1) {
    const [prev, cur] = [data.folds[i - 1].train, data.folds[i].train];
    assert.equal(cur.length, prev.length + 10);
    assert.ok(prev.every((t) => cur.includes(t)) && cur.length > prev.length, "strict superset");
  }
  assert.deepEqual(V.temporalFolds(), data, "expanding is the default");
});

test("temporalFolds sliding: fixed width 30 advances by 10 and drops the oldest history", () => {
  const data = V.temporalFolds({ kind: "sliding" });
  assert.equal(data.kind, "sliding");
  assert.deepEqual(data.folds.map((f) => [f.train, f.validation]), SLIDING);
  assertTemporal(data);
  data.folds.forEach((f, i) => {
    assert.equal(f.train.length, 30, "constant train width");
    assert.deepEqual(f.discarded, range(0, 10 * i), "old history discarded, not unused future");
  });
  const expanding = V.temporalFolds({ kind: "expanding" });
  data.folds.forEach((f, i) => assert.deepEqual(f.validation, expanding.folds[i].validation, "same validation blocks"));
});

test("temporalFolds validates parameters and never mutates inputs", () => {
  const options = Object.freeze({ kind: "sliding", initialTrain: 30, horizon: 10, step: 10, folds: 5 });
  const a = V.temporalFolds(options);
  assert.deepEqual(options, { kind: "sliding", initialTrain: 30, horizon: 10, step: 10, folds: 5 });
  a.folds[0].train.length = 0;
  a.test.push(-1);
  assert.deepEqual(V.temporalFolds(options).folds[0].train, range(0, 30), "fresh copies every call");
  assert.deepEqual(V.temporalFolds(options).test, FINAL_BLOCK);
  const custom = V.temporalFolds({ kind: "expanding", initialTrain: 20, horizon: 5, step: 15, folds: 3 });
  assert.deepEqual(custom.folds.map((f) => [f.train.length, f.validation]), [
    [20, range(20, 25)],
    [35, range(35, 40)],
    [50, range(50, 55)],
  ]);
  assert.deepEqual(custom.test, FINAL_BLOCK);
  assert.throws(() => V.temporalFolds({ kind: "random" }), /kind/);
  assert.throws(() => V.temporalFolds({ horizon: 0 }), /horizon/);
  assert.throws(() => V.temporalFolds({ step: 1.5 }), /step/);
  assert.throws(() => V.temporalFolds({ initialTrain: -2 }), /initialTrain/);
  assert.throws(() => V.temporalFolds({ folds: 6 }), /final test/, "a sixth fold would validate on t = 80..89");
  assert.throws(() => V.temporalFolds({ kind: "sliding", trainWidth: 0 }), /trainWidth/);
});

test("foldRoles labels every time index once; holdout is always test", () => {
  const exp = V.temporalFolds({ kind: "sliding" }).folds[2];
  const roles = V.foldRoles(exp);
  assert.equal(roles.length, 100);
  assert.deepEqual(times(roles.map((r, t) => ({ t, set: r })), "discarded"), range(0, 20));
  assert.deepEqual(times(roles.map((r, t) => ({ t, set: r })), "train"), range(20, 50));
  assert.deepEqual(times(roles.map((r, t) => ({ t, set: r })), "validation"), range(50, 60));
  assert.deepEqual(times(roles.map((r, t) => ({ t, set: r })), "unused"), range(60, 80));
  assert.deepEqual(times(roles.map((r, t) => ({ t, set: r })), "test"), FINAL_BLOCK);
  const iid = V.foldRoles(V.iidFolds().folds[0]);
  assert.deepEqual(times(iid.map((r, t) => ({ t, set: r })), "test"), HOLDOUT);
});

// ---------- Rendered frames ----------

function assertFrameBasics(out) {
  assert.match(out.svg, /^<svg[^>]*viewBox="0 0 600 440"/);
  assert.ok(out.label.length > 10);
  assert.match(out.svg, /data-role="lock"/, "final test drawn as locked");
}

test("folds lesson: each frame plots exactly the helper's fold on the original axis, test locked outside", () => {
  const lesson = V.LESSONS.folds;
  const data = V.iidFolds();
  assert.equal(lesson.frameCount, data.folds.length + 2);
  for (let frame = 0; frame < lesson.frameCount; frame += 1) {
    const out = lesson.render(frame);
    assertFrameBasics(out);
    const main = points(out.svg, "main");
    assert.equal(main.length, 100);
    assertCanonical(main);
    assert.deepEqual(times(main, "test"), HOLDOUT, `frame ${frame}: holdout visible and locked`);
    const k = frame - 1;
    if (k >= 0 && k < 5) {
      const fold = data.folds[k];
      assert.deepEqual(times(main, "train"), fold.train);
      assert.deepEqual(times(main, "validation"), fold.validation);
      assert.equal(count(out.svg, "train"), 64);
      assert.equal(count(out.svg, "validation"), 16);
      assert.equal(count(out.svg, "test"), 20);
      assert.match(out.svg, new RegExp(`data-current-fold="${k}"`));
      assert.ok(out.label.includes(`${k + 1}/5`), out.label);
    }
    // The student sees every fold row in every frame, each exactly its helper roles.
    const rowCells = cells(out.svg, "fold");
    data.folds.forEach((fold, i) => {
      const row = rowCells.filter((c) => c.fold === i);
      assert.deepEqual(row.map((c) => c.role), V.foldRoles(fold), `frame ${frame} row ${i}`);
    });
  }
  assert.deepEqual(times(points(lesson.render(0).svg, "main"), "development"), DEV);
  const summary = cells(lesson.render(6).svg, "fold").filter((c) => c.role === "validation");
  assert.deepEqual(sorted(summary.map((c) => c.t)), DEV, "summary: every development sample validated once");
  assert.match(lesson.render(1).svg, /IID/);
  assert.match(lesson.render(6).svg, /no se usa para elegir/);
});

test("leakage lesson: a later training point next to an earlier validation point, no fake scores", () => {
  const lesson = V.LESSONS.leakage;
  const fold = V.iidFolds().folds[V.LEAKAGE_FOLD];
  const ex = V.leakageExample();
  assert.ok(ex.futureTrain > ex.validation, "the training point is in the future of the validation point");
  assert.ok(fold.validation.includes(ex.validation));
  assert.ok(fold.train.includes(ex.futureTrain), "future point is training data");
  assert.ok(!HOLDOUT.includes(ex.futureTrain), "never the locked final test");
  assert.equal(ex.gap, ex.futureTrain - ex.validation);
  assert.ok(ex.gap >= 1 && ex.gap <= 3, "close in time, so autocorrelated");

  for (let frame = 0; frame < lesson.frameCount; frame += 1) {
    const out = lesson.render(frame);
    assertFrameBasics(out);
    const main = points(out.svg, "main");
    assertCanonical(main);
    assert.deepEqual(times(main, "train"), fold.train);
    assert.deepEqual(times(main, "validation"), fold.validation);
    assert.deepEqual(times(main, "test"), HOLDOUT);
    assert.match(out.svg, /data-role="reference"/);
    assert.doesNotMatch(out.svg, /<path[^>]*data-plot=/, "no lines across missing points");
  }
  const shapes = lesson.render(1).svg;
  assert.deepEqual(times(points(shapes, "train-only"), "train"), fold.train);
  assert.deepEqual(times(points(shapes, "validation-only"), "validation"), fold.validation);
  assertCanonical(points(shapes, "train-only").concat(points(shapes, "validation-only")));

  const leak = lesson.render(2);
  assert.match(leak.svg, new RegExp(`data-role="leak-arrow" data-from="${ex.futureTrain}" data-to="${ex.validation}"`));
  assert.ok(leak.label.includes(`t = ${ex.futureTrain}`) && leak.label.includes(`t = ${ex.validation}`), leak.label);
  const why = lesson.render(3);
  assert.ok(why.svg.includes(T.SIGNAL[ex.validation].y.toFixed(2)) && why.svg.includes(T.SIGNAL[ex.futureTrain].y.toFixed(2)));
  const all = range(0, lesson.frameCount).map((f) => lesson.render(f).svg).join("");
  assert.match(all, /autocorrelación/);
  assert.match(all, /no prueba/);
  assert.match(all, /no siempre está prohibida/);
  assert.match(all, /no se ajusta ningún modelo/);
  assert.doesNotMatch(all, /accuracy|precisión de|R² =|MSE =/, "no invented performance numbers");
  const contrast = lesson.render(4).svg;
  assert.deepEqual(times(cells(contrast, "chronological"), "train", "role"), range(0, 30));
  assert.deepEqual(times(cells(contrast, "chronological"), "validation", "role"), range(30, 40));
  assert.deepEqual(times(cells(contrast, "random"), "validation", "role"), fold.validation);
});

function assertTemporalLesson(kind) {
  const lesson = V.LESSONS[kind];
  const data = V.temporalFolds({ kind });
  assert.equal(lesson.frameCount, 1 + data.folds.length * 3 + 1);
  for (let frame = 0; frame < lesson.frameCount; frame += 1) {
    const out = lesson.render(frame);
    assertFrameBasics(out);
    const main = points(out.svg, "main");
    assert.equal(main.length, 100);
    assertCanonical(main);
    assert.deepEqual(times(main, "test"), FINAL_BLOCK, `frame ${frame}: final test locked`);
    assert.equal(count(out.svg, "test"), 20);
    // All fold rows are drawn, each exactly its helper roles.
    const rowCells = cells(out.svg, "fold");
    data.folds.forEach((fold, i) => {
      const row = rowCells.filter((c) => c.fold === i);
      assert.deepEqual(row.map((c) => c.role), V.foldRoles(fold), `${kind} row ${i}`);
    });
    if (frame === 0 || frame === lesson.frameCount - 1) continue;
    const k = Math.floor((frame - 1) / 3);
    const fold = data.folds[k];
    const roles = V.foldRoles(fold);
    assert.deepEqual(main.map((p) => p.set), roles, `${kind} frame ${frame}: plotted roles = fold ${k}`);
    assert.equal(count(out.svg, "train"), fold.train.length);
    assert.equal(count(out.svg, "validation"), 10);
    assert.equal(count(out.svg, "unused"), fold.unused.length);
    assert.match(out.svg, new RegExp(`data-current-fold="${k}"`));
    assert.match(out.svg, new RegExp(`data-origin="${fold.origin}"`));
    assert.ok(out.label.includes(`${k + 1}/5`), out.label);
    assert.ok(out.label.includes(`${fold.train[0]}…${fold.train[fold.train.length - 1]}`), out.label);
  }
  const intro = lesson.render(0).svg;
  assert.deepEqual(times(points(intro, "main"), "development"), range(0, 80));
  return lesson;
}

test("expanding lesson: fit → validate → advance per fold, counts and origin from temporalFolds", () => {
  const lesson = assertTemporalLesson("expanding");
  const labels = [1, 2, 3].map((f) => lesson.render(f).label);
  assert.match(labels[0], /ajusta/);
  assert.match(labels[1], /valida/);
  assert.match(labels[2], /avanza/);
  assert.equal(count(lesson.render(1).svg, "train"), 30);
  assert.equal(count(lesson.render(13).svg, "train"), 70);
  assert.match(lesson.render(16).svg, /TimeSeriesSplit/);
});

test("sliding lesson: discarded history differs from unused future; expanding comparison row", () => {
  const lesson = assertTemporalLesson("sliding");
  const expanding = V.temporalFolds({ kind: "expanding" });
  for (let k = 0; k < 5; k += 1) {
    const svg = lesson.render(1 + 3 * k).svg;
    assert.equal(count(svg, "train"), 30);
    assert.equal(count(svg, "discarded"), 10 * k);
    const cmp = cells(svg, "compare");
    assert.deepEqual(cmp.map((c) => c.role), V.foldRoles(expanding.folds[k]), "comparison row = expanding fold");
  }
  const discarded = /data-set="discarded"[^>]*fill="([^"]+)"/.exec(lesson.render(13).svg)[1];
  const unused = /data-set="unused"[^>]*fill="([^"]+)"/.exec(lesson.render(4).svg)[1];
  assert.notEqual(discarded, unused, "discarded history and unused future look different");
});

// ---------- Mount with a minimal fake DOM ----------

function fakeDom({ reduced = false } = {}) {
  const timers = new Map();
  let id = 0;
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
  doc.defaultView = {
    setInterval(fn) {
      id += 1;
      timers.set(id, fn);
      return id;
    },
    clearInterval(t) {
      timers.delete(t);
    },
    matchMedia: () => ({ matches: reduced, addEventListener() {} }),
    MutationObserver: MO,
  };
  doc.createElement = (t) => new El(t);
  doc.querySelectorAll = (sel) => doc.mounts.filter((el) => el.getAttribute(sel.slice(1, -1)) !== null);
  const tick = () => [...timers.values()].forEach((fn) => fn());
  return { doc, El, timers, tick, notify: () => observers.forEach((o) => o.cb([])) };
}

function mountOn(dom, kind, active) {
  const slide = new dom.El("section");
  slide.classList.add("slide");
  if (active) slide.classList.add("is-active");
  const el = new dom.El("div");
  el.setAttribute("data-validation-lesson", kind);
  slide.appendChild(el);
  dom.doc.mounts.push(el);
  return { slide, el };
}

const keyEvent = (key, target) => ({
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
});

test("autoMount builds the four lessons, plays only on the active slide and is idempotent", () => {
  const dom = fakeDom();
  const kinds = ["folds", "leakage", "expanding", "sliding"];
  const slides = kinds.map((kind) => mountOn(dom, kind, false));
  const other = new dom.El("div");
  other.setAttribute("data-validation-lesson", "random");
  dom.doc.mounts.push(other);
  const instances = V.autoMount(dom.doc);
  assert.equal(instances.length, 4);
  instances.forEach((inst, i) => {
    assert.equal(inst.kind, kinds[i]);
    assert.ok(inst.root.classList.contains("validation-lesson"));
    assert.equal(inst.stage.innerHTML, V.LESSONS[kinds[i]].render(0).svg);
    assert.equal(inst.label.getAttribute("aria-live"), "polite");
    assert.equal(inst.playButton.textContent, "Pausar");
  });
  assert.equal(dom.timers.size, 0, "inactive slides: no timers");
  assert.equal(V.autoMount(dom.doc)[3], instances[3], "mount is idempotent");
  assert.equal(slides[3].el.children.length, 3, "no duplicated controls");

  slides.forEach((s) => s.slide.classList.add("is-active"));
  dom.notify();
  assert.equal(dom.timers.size, 4);
  dom.tick();
  instances.forEach((inst, i) => assert.equal(inst.stage.innerHTML, V.LESSONS[kinds[i]].render(1).svg));
  slides.forEach((s) => s.slide.classList.remove("is-active"));
  dom.notify();
  assert.equal(dom.timers.size, 0, "hidden slides clear timers");
  assert.throws(() => V.mount(new dom.El("div"), "shuffle"), /Unknown validation lesson/);
});

test("reduced motion mounts paused; manual steps and widget-scoped keys still work", () => {
  const dom = fakeDom({ reduced: true });
  const { el } = mountOn(dom, "sliding", true);
  const inst = V.mount(el);
  assert.equal(dom.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst.player.state.frame, 1);
  assert.equal(inst.stage.innerHTML, V.LESSONS.sliding.render(1).svg);
  const end = keyEvent("End", el);
  el.dispatch("keydown", end);
  assert.equal(end.stopped, true);
  assert.equal(inst.player.state.frame, V.LESSONS.sliding.frameCount - 1);
  const f = keyEvent("f", el);
  el.dispatch("keydown", f);
  assert.equal(f.stopped, false, "fullscreen key still reaches the deck");
  assert.equal(dom.timers.size, 0);
});
