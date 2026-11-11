"use strict";

// Deterministic checks for the im2col matrix lesson (node --test).
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const dir = path.join(__dirname, "..", "docs", "pregrado", "animations");
const S = require(path.join(dir, "spatial.js"));
const M = require(path.join(dir, "matrix.js"));

const HOUSE = [
  [0, 0, 1, 0, 0],
  [0, 1, 0, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 1, 1, 0],
];
const VERTICAL_COL = [0, 1, 0, 0, 1, 0, 0, 1, 0];
const HORIZONTAL_COL = [0, 0, 0, 1, 1, 1, 0, 0, 0];

function cellValues(svg, prefix) {
  const re = new RegExp(`data-cell="${prefix}-(\\d+)-(\\d+)"[^>]*data-value="(-?[\\d.]+)"`, "g");
  const out = {};
  for (const m of svg.matchAll(re)) out[`${m[1]}-${m[2]}`] = Number(m[3]);
  return out;
}

const productFrame = (f, p) => M.PHASES.products + f * 9 + p;

// ---------- Pure helpers ----------

test("matrix lesson reuses the canonical spatial data", () => {
  assert.equal(M.HOUSE, S.HOUSE);
  assert.equal(M.FILTER_BANK, S.FILTER_BANK);
  assert.deepEqual(M.HOUSE, HOUSE);
});

test("X_patches is 9x9: row-major windows, each flattened row-major", () => {
  const X = M.patchMatrix(HOUSE, 3);
  assert.equal(X.length, 9);
  X.forEach((row) => assert.equal(row.length, 9));
  assert.deepEqual(X[0], [0, 0, 1, 0, 1, 0, 1, 1, 1]);
  assert.deepEqual(X[1], [0, 1, 0, 1, 0, 1, 1, 1, 1]);
  assert.deepEqual(X[3], [0, 1, 0, 1, 1, 1, 0, 1, 0]);
  assert.deepEqual(X[4], [1, 0, 1, 1, 1, 1, 1, 0, 1]);
  assert.deepEqual(X[8], [1, 1, 1, 0, 1, 0, 1, 1, 0]);
  for (let p = 0; p < 9; p += 1) {
    assert.deepEqual(X[p], S.windowAt(HOUSE, Math.floor(p / 3), p % 3, 3).flat());
  }
});

test("overlapping windows repeat the same pixel in several rows", () => {
  const X = M.patchMatrix(HOUSE, 3);
  // Pixel (0,1) is entry 1 of window 1 and entry 0 of window 2.
  assert.equal(X[0][1], HOUSE[0][1]);
  assert.equal(X[1][0], HOUSE[0][1]);
  // The center pixel (2,2) lands in all nine rows, at a different column each time.
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) assert.equal(X[r * 3 + c][(2 - r) * 3 + (2 - c)], HOUSE[2][2]);
  }
});

test("W is 9x3: each flattened filter is a column, not a row", () => {
  const W = M.filterMatrix(S.FILTER_BANK);
  assert.equal(W.length, 9);
  W.forEach((row) => assert.equal(row.length, 3));
  assert.deepEqual(W.map((row) => row[0]), VERTICAL_COL);
  assert.deepEqual(W.map((row) => row[1]), HORIZONTAL_COL);
  assert.deepEqual(W.map((row) => row[2]), VERTICAL_COL, "duplicate filter gives a duplicate column");
  assert.deepEqual(W[4], [1, 1, 1]);
  assert.deepEqual(W[1], [1, 0, 1]);
});

test("Y = X_patches · W (9x9 · 9x3 = 9x3) equals direct convolution for all 27 products", () => {
  const { X, W, Y, maps } = M.matrixData();
  assert.deepEqual(M.shape(X), [9, 9]);
  assert.deepEqual(M.shape(W), [9, 3]);
  assert.deepEqual(M.shape(Y), [9, 3]);
  assert.deepEqual(M.matmul(X, W), Y);
  assert.throws(() => M.matmul(W, W), /shape/);
  let checked = 0;
  S.FILTER_BANK.forEach((filter, f) => {
    const direct = S.convolve2d(HOUSE, filter.kernel);
    for (let p = 0; p < 9; p += 1) {
      assert.equal(Y[p][f], direct[Math.floor(p / 3)][p % 3], `Y[${p}][${f}]`);
      checked += 1;
    }
    assert.deepEqual(maps[f], direct);
    assert.deepEqual(M.columnToMap(Y, f, 3), direct);
  });
  assert.equal(checked, 27);
  assert.deepEqual(Y.map((row) => row[2]), Y.map((row) => row[0]), "redundant columns");
  assert.notDeepEqual(Y.map((row) => row[1]), Y.map((row) => row[0]));
});

// ---------- Rendered frames ----------

test("every frame is a 600x440 SVG that shows the exact shapes", () => {
  const lesson = M.LESSON;
  assert.equal(lesson.frameCount, 40);
  assert.equal(S.LESSONS.matrix, lesson, "registered for the shared mount");
  for (let i = 0; i < lesson.frameCount; i += 1) {
    const f = lesson.render(i);
    assert.match(f.svg, /^<svg[^>]*viewBox="0 0 600 440"/);
    for (const s of ["X_patches (9×9)", "W (9×3)", "Y (9×3)", "3 mapas 3×3"]) assert.ok(f.svg.includes(s), `${i}: ${s}`);
    assert.ok(f.label.length > 0);
    assert.doesNotMatch(f.svg + f.label, /NaN|undefined/);
  }
});

test("window frames flatten each colored patch into the next X row", () => {
  const X = M.patchMatrix(HOUSE, 3);
  for (let p = 0; p < 9; p += 1) {
    const f = M.LESSON.render(p);
    const x = cellValues(f.svg, "x");
    assert.equal(Object.keys(x).length, (p + 1) * 9, "rows appear one by one");
    for (let i = 0; i < 9; i += 1) assert.equal(x[`${p}-${i}`], X[p][i]);
    assert.deepEqual(Object.values(cellValues(f.svg, "patch")), X[p]);
    assert.equal(Object.keys(cellValues(f.svg, "y")).length, 0);
    assert.match(f.label, new RegExp(`Ventana ${p + 1}/9`));
    assert.match(f.label, new RegExp(`fila ${p + 1} de X`));
  }
  assert.match(M.LESSON.render(1).svg + M.LESSON.render(1).label, /solapan/);
});

test("filter frames turn each kernel into a column of W", () => {
  for (let f = 0; f < 3; f += 1) {
    const frame = M.LESSON.render(M.PHASES.filters + f);
    const w = cellValues(frame.svg, "w");
    assert.equal(Object.keys(w).length, (f + 1) * 9);
    const col = Array.from({ length: 9 }, (_, i) => w[`${i}-${f}`]);
    assert.deepEqual(col, f === 1 ? HORIZONTAL_COL : VERTICAL_COL);
    assert.deepEqual(Object.values(cellValues(frame.svg, "kernel")), S.FILTER_BANK[f].kernel.flat());
    assert.equal(Object.keys(cellValues(frame.svg, "x")).length, 81, "X stays complete");
    assert.match(frame.label, new RegExp(`columna ${f + 1} de W`));
  }
});

test("product frames show the dot product, the Y cell and the map cell with real values", () => {
  const { X, W, Y } = M.matrixData();
  for (let f = 0; f < 3; f += 1) {
    for (let p = 0; p < 9; p += 1) {
      const frame = M.LESSON.render(productFrame(f, p));
      const y = cellValues(frame.svg, "y");
      const map = cellValues(frame.svg, `m${f}`);
      const r = Math.floor(p / 3);
      const c = p % 3;
      assert.equal(Object.keys(y).length, f * 9 + p + 1, "Y fills column by column");
      assert.equal(y[`${p}-${f}`], Y[p][f]);
      assert.equal(map[`${r}-${c}`], Y[p][f]);
      assert.equal(Object.keys(map).length, p + 1);
      const terms = X[p].map((x, i) => `${x}·${W[i][f]}`).join(" + ");
      assert.ok(frame.svg.replace(/<[^>]+>/g, "").includes(`${terms} = ${Y[p][f]}`), `terms ${f}/${p}`);
      assert.match(frame.label, new RegExp(`fila ${p + 1} de X · columna ${f + 1} de W = ${Y[p][f]}`));
      assert.match(frame.label, new RegExp(`mapa ${f + 1}\\[${r + 1}, ${c + 1}\\]`));
    }
  }
});

test("final frame reshapes Y into three maps and flags the repeated filter", () => {
  const { maps } = M.matrixData();
  const last = M.LESSON.render(M.LESSON.frameCount - 1);
  assert.equal(Object.keys(cellValues(last.svg, "y")).length, 27);
  for (let f = 0; f < 3; f += 1) {
    assert.deepEqual(Object.values(cellValues(last.svg, `m${f}`)), maps[f].flat());
  }
  const all = last.svg + last.label;
  assert.match(all, /redundancia/);
  assert.match(all, /XW/);
  assert.match(all, /densa/);
  assert.match(all, /im2col/);
  assert.match(all, /sin copiar/, "does not claim every backend allocates the windows");
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
  doc.querySelectorAll = (sel) => (sel === "[data-matrix-lesson]" ? doc.mounts : []);
  const notify = () => observers.forEach((o) => o.cb([]));
  const lessonOnSlide = (active) => {
    const slide = new El("section");
    slide.classList.add("slide");
    if (active) slide.classList.add("is-active");
    const el = new El("div");
    el.setAttribute("data-matrix-lesson", "");
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

test("autoMount builds the lesson and plays only on the active, visible slide", () => {
  const dom = fakeDom();
  const { slide } = dom.lessonOnSlide(false);
  const [inst] = M.autoMount(dom.doc);
  assert.ok(inst.root.classList.contains("matrix-lesson"));
  assert.match(inst.stage.innerHTML, /^<svg[^>]*viewBox="0 0 600 440"/);
  assert.equal(inst.label.getAttribute("aria-live"), "polite");
  assert.equal(inst.playButton.textContent, "Pausar");
  assert.equal(inst.nextButton.textContent, "Paso siguiente");
  assert.equal(dom.sched.timers.size, 0, "inactive slide: no timer");
  assert.equal(M.autoMount(dom.doc)[0], inst, "mount is idempotent");

  slide.classList.add("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 1);
  dom.sched.tick();
  assert.equal(inst.player.state.frame, 1);
  assert.match(inst.label.textContent, /Ventana 2\/9/);

  dom.doc.hidden = true;
  dom.doc.fire("visibilitychange");
  assert.equal(dom.sched.timers.size, 0, "hidden tab clears the timer");
  dom.doc.hidden = false;
  dom.doc.fire("visibilitychange");
  assert.equal(dom.sched.timers.size, 1);
  slide.classList.remove("is-active");
  dom.notify();
  assert.equal(dom.sched.timers.size, 0, "leaving the slide clears the timer");
  inst.destroy();
});

test("Play/Pause and Step buttons; deck keys stay scoped to the widget", () => {
  const dom = fakeDom();
  const { el } = dom.lessonOnSlide(true);
  const inst = M.mount(el);
  assert.equal(dom.sched.timers.size, 1);
  inst.playButton.dispatch("click", { detail: 1 });
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.playButton.dispatch("click", { detail: 1 });
  assert.equal(dom.sched.timers.size, 1);
  inst.nextButton.dispatch("click", { detail: 1 });
  assert.equal(inst.player.state.playing, false, "manual step pauses");
  assert.equal(inst.player.state.frame, 1);

  const end = key("End", el);
  el.dispatch("keydown", end);
  assert.equal(end.stopped, true);
  assert.equal(inst.player.state.frame, M.LESSON.frameCount - 1);
  assert.match(inst.label.textContent, /redundancia/);
  el.dispatch("keydown", key("ArrowRight", el));
  assert.equal(inst.player.state.frame, 0, "wraps around");
  const other = key("f", el);
  el.dispatch("keydown", other);
  assert.equal(other.stopped, false, "non-navigation keys reach the deck");
  inst.destroy();
});

test("reduced motion mounts paused but manual step works", () => {
  const dom = fakeDom({ reduced: true });
  const { el } = dom.lessonOnSlide(true);
  const inst = M.mount(el);
  assert.equal(dom.sched.timers.size, 0);
  assert.equal(inst.playButton.textContent, "Reproducir");
  inst.nextButton.dispatch("click", { detail: 0 });
  assert.equal(inst.player.state.frame, 1);
  assert.match(inst.label.textContent, /Ventana 2\/9/);
});
