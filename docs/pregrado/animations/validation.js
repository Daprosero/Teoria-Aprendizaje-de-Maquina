/*
 * Validation lessons: ordinary K-fold with a fixed external test, why a random split misleads forecasting,
 * expanding-window and fixed-width sliding-window temporal cross-validation.
 * Mount point: [data-validation-lesson="folds" | "leakage" | "expanding" | "sliding"].
 * Requires spatial.js and temporal.js first (browser globals SpatialLesson and TemporalLesson,
 * or require("./spatial.js") / require("./temporal.js") in Node): the player, controls, keyboard
 * handling, canonical signal, partition helper and SVG primitives all come from there.
 * Browser: global ValidationLesson (auto-mounts). Node: CommonJS exports for tests.
 * Every fold index and count drawn comes from iidFolds / temporalFolds, the same helpers the tests use.
 */
(function (root, factory) {
  const load = (name, file) =>
    root && root[name] ? root[name] : typeof require === "function" ? require(file) : null;
  const spatial = load("SpatialLesson", "./spatial.js");
  const temporal = load("TemporalLesson", "./temporal.js");
  if (!spatial || !temporal) throw new Error("ValidationLesson needs spatial.js and temporal.js loaded first");
  const api = factory(spatial, temporal);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.ValidationLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function (S, T) {
  "use strict";

  const { COLORS } = S;
  const { SIGNAL, SIGNAL_LENGTH, Y_DOMAIN, ROLE_COLORS } = T;
  const { text, line, arrow, svg, timeScale, axes, seriesPath, dots, band, centered } = T.plot;

  // Fill per role. Discarded history (light) and unused future (darker gray) must stay distinguishable.
  const ROLE_FILL = Object.freeze({
    train: ROLE_COLORS.train,
    validation: ROLE_COLORS.validation,
    test: ROLE_COLORS.test,
    unused: ROLE_COLORS.pending,
    discarded: "#D3D9E0",
    development: COLORS.muted,
  });
  const ROLE_NAMES = Object.freeze({
    train: "train",
    validation: "validación",
    test: "test final",
    unused: "sin usar",
    discarded: "descartado",
    development: "desarrollo",
  });

  const IID_DEFAULTS = Object.freeze({ k: 5, seed: 7, holdoutSeed: 7, testFraction: 0.2 });
  const TEMPORAL_DEFAULTS = Object.freeze({ kind: "expanding", initialTrain: 30, horizon: 10, step: 10, folds: 5, testFraction: 0.2 });
  const LEAKAGE_FOLD = 0;

  const r1 = (v) => Math.round(v * 10) / 10;
  const fmt = (v) => v.toFixed(2);
  const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
  const span = (idx) => `${idx[0]}…${idx[idx.length - 1]}`;

  function assertCount(name, value, min) {
    if (!Number.isInteger(value) || value < min) throw new RangeError(`${name} must be an integer ≥ ${min}`);
  }

  // ---------- Ordinary K-fold inside a development pool ----------

  /*
   * One random external holdout (partition random, 20 %, seed 7) is set aside ONCE. The remaining
   * development indices are shuffled with `seed` and cut into k validation groups; each fold trains
   * on the rest of the development pool. Returns fresh sorted arrays; the holdout is identical in all folds.
   */
  function iidFolds(options) {
    const o = Object.assign({}, IID_DEFAULTS, options);
    const part = T.partition({ mode: "random", testFraction: o.testFraction, seed: o.holdoutSeed });
    const dev = part.train;
    if (!Number.isInteger(o.k) || o.k < 2 || o.k > dev.length) {
      throw new RangeError(`k must be an integer between 2 and ${dev.length}`);
    }
    const shuffled = T.seededShuffle(dev, o.seed);
    const base = Math.floor(dev.length / o.k);
    const extra = dev.length % o.k;
    const byTime = (a, b) => a - b;
    let start = 0;
    const folds = Array.from({ length: o.k }, (_, index) => {
      const size = base + (index < extra ? 1 : 0);
      const validation = shuffled.slice(start, start + size).sort(byTime);
      start += size;
      const inValidation = new Set(validation);
      return { index, train: dev.filter((t) => !inValidation.has(t)), validation, test: part.test.slice() };
    });
    return { holdout: part.test.slice(), development: dev.slice(), folds };
  }

  // ---------- Temporal folds: forecast origin moves forward, final block untouched ----------

  /*
   * Validation fold i covers [origin, origin + horizon), origin = initialTrain + i·step.
   * expanding: train = every index before the origin (all the past is kept).
   * sliding:   train = the trainWidth indices just before the origin; older ones are "discarded".
   * The final chronological test block (last 20 %) is never inside any fold.
   */
  function temporalFolds(options) {
    const o = Object.assign({}, TEMPORAL_DEFAULTS, options);
    if (o.kind !== "expanding" && o.kind !== "sliding") throw new RangeError('kind must be "expanding" or "sliding"');
    assertCount("initialTrain", o.initialTrain, 1);
    assertCount("horizon", o.horizon, 1);
    assertCount("step", o.step, 1);
    assertCount("folds", o.folds, 1);
    const width = o.kind === "sliding" ? (o.trainWidth == null ? o.initialTrain : o.trainWidth) : null;
    if (o.kind === "sliding") {
      assertCount("trainWidth", width, 1);
      if (width > o.initialTrain) throw new RangeError("trainWidth must be ≤ initialTrain");
    }
    const part = T.partition({ mode: "chronological", testFraction: o.testFraction });
    const testStart = part.test[0];
    const lastEnd = o.initialTrain + (o.folds - 1) * o.step + o.horizon;
    if (lastEnd > testStart) throw new RangeError(`folds must end before the final test block (t = ${testStart})`);
    const folds = Array.from({ length: o.folds }, (_, index) => {
      const origin = o.initialTrain + index * o.step;
      const from = o.kind === "sliding" ? origin - width : 0;
      return {
        index,
        origin,
        discarded: range(0, from),
        train: range(from, origin),
        validation: range(origin, origin + o.horizon),
        unused: range(origin + o.horizon, testStart),
        test: part.test.slice(),
      };
    });
    return {
      kind: o.kind,
      initialTrain: o.initialTrain,
      horizon: o.horizon,
      step: o.step,
      trainWidth: width,
      development: part.train.slice(),
      test: part.test.slice(),
      folds,
    };
  }

  // Role of every time index in one fold (IID or temporal): discarded, train, validation, unused or test.
  function foldRoles(fold, n) {
    const roles = Array.from({ length: n == null ? SIGNAL_LENGTH : n }, () => "unused");
    [["discarded", fold.discarded || []], ["train", fold.train], ["validation", fold.validation], ["test", fold.test]].forEach(
      ([role, idx]) => idx.forEach((t) => (roles[t] = role))
    );
    return roles;
  }

  // A validation time in the random fold with a training time 1–3 steps later (never the holdout).
  function leakageExample(foldIndex) {
    const index = foldIndex == null ? LEAKAGE_FOLD : foldIndex;
    const fold = iidFolds().folds[index];
    const train = new Set(fold.train);
    for (let gap = 1; gap <= 3; gap += 1) {
      const v = fold.validation.find((t) => t >= 20 && t < 70 && train.has(t + gap));
      if (v != null) return { fold: index, validation: v, futureTrain: v + gap, gap };
    }
    throw new Error("no nearby future training point for this fold");
  }

  // ---------- Shared drawing ----------

  const IID = iidFolds();
  const TEMPORAL = { expanding: temporalFolds({ kind: "expanding" }), sliding: temporalFolds({ kind: "sliding" }) };
  const LEAK = leakageExample();
  const MAIN_BOX = Object.freeze({ x: 60, y: 70, width: 500, height: 110 });
  // Temporal lessons keep a strip above the plot for the origin and stage marks, clear of the points.
  const TEMPORAL_BOX = Object.freeze({ x: 60, y: 86, width: 500, height: 94 });

  function lock(x, y, color) {
    const c = color || ROLE_COLORS.test;
    return (
      `<g data-role="lock"><path d="M ${r1(x - 3.5)} ${y} v -3 a 3.5 3.5 0 0 1 7 0 v 3" fill="none" stroke="${c}" stroke-width="1.6"/>` +
      `<rect x="${r1(x - 5)}" y="${y}" width="10" height="8" rx="1.5" fill="${c}"/></g>`
    );
  }

  // Signal on its original time axis: faint reference curve plus one dot per t colored by role.
  function mainPlot(roles, ticks, box) {
    const sc = timeScale(box || MAIN_BOX);
    let b = axes(sc, { ticks: ticks || [0, 20, 40, 60, 80, 99], yTicks: [Y_DOMAIN[0], Y_DOMAIN[1]] });
    b += seriesPath(SIGNAL, sc, { color: ROLE_COLORS.reference, width: 1.2, opacity: 0.35, role: "reference" });
    b += dots(SIGNAL, sc, { set: (p) => roles[p.t], fill: (p) => ROLE_FILL[roles[p.t]], r: 3 });
    return { sc, b };
  }

  // One cell per time index, aligned with the main plot; the current fold gets an outline.
  function foldRow(sc, y, roles, o) {
    const w = sc.width / (sc.tMax - sc.tMin);
    let b = `<g opacity="${o.dim ? 0.55 : 1}">`;
    roles.forEach((role, t) => {
      b +=
        `<rect data-row="${o.row}" data-fold="${o.fold}" data-t="${t}" data-role="${role}"` +
        ` x="${r1(sc.sx(t) - w / 2)}" y="${y}" width="${r1(w)}" height="11" fill="${ROLE_FILL[role]}"/>`;
    });
    if (o.label) b += text(sc.x - 8, y + 10, o.label, { size: 10.5, anchor: "end", fill: COLORS.muted, weight: "bold" });
    b += "</g>";
    if (o.current) {
      b +=
        `<rect data-current-fold="${o.fold}" x="${r1(sc.x - w / 2 - 2)}" y="${y - 2}" width="${r1(sc.width + w + 4)}"` +
        ` height="15" rx="2" fill="none" stroke="${COLORS.ink}" stroke-width="1.5"/>`;
    }
    return b;
  }

  function foldRows(sc, y, folds, current) {
    return folds
      .map((fold, i) =>
        foldRow(sc, y + i * 17, foldRoles(fold), { row: "fold", fold: i, label: `P${i + 1}`, current: i === current, dim: current != null && i !== current })
      )
      .join("");
  }

  // Counts as colored text, evenly spaced; each value is the length of an actual index set.
  function countsRow(items, y) {
    const gap = Math.min(140, 560 / items.length);
    return items
      .map((it, i) => {
        const x = r1(300 + (i - (items.length - 1) / 2) * gap);
        const label = `${ROLE_NAMES[it.role]}: ${it.value}`;
        const lockMark = it.role === "test" ? lock(r1(x - label.length * 3.4 - 10), y - 9) : "";
        return (
          `<g data-count="${it.role}" data-value="${it.value}">${lockMark}` +
          `${text(x, y, label, { size: 12, anchor: "middle", weight: "bold", fill: ROLE_FILL[it.role] })}</g>`
        );
      })
      .join("");
  }

  const header = (title, subtitle) =>
    text(300, 26, title, { size: 16.5, weight: "bold", anchor: "middle" }) +
    text(300, 46, subtitle, { size: 12.5, anchor: "middle", fill: COLORS.muted });

  // ---------- Lesson 1: ordinary K-fold, final test outside every fold ----------

  const FOLDS_FRAMES = IID.folds.length + 2;

  function renderFolds(frame) {
    const k = frame - 1;
    const summary = frame === FOLDS_FRAMES - 1;
    const fold = k >= 0 && !summary ? IID.folds[k] : null;
    const devRoles = foldRoles({ train: IID.development, validation: [], test: IID.holdout }).map((r) => (r === "train" ? "development" : r));
    const { sc, b: plot } = mainPlot(fold ? foldRoles(fold) : devRoles);
    const subtitle = fold
      ? `Pliegue ${k + 1}/5: validación (ámbar) y train (azul) dentro del desarrollo`
      : summary
      ? "Cada punto de desarrollo fue validación exactamente una vez"
      : "Primero se aparta el test final: 20 puntos al azar (semilla 7)";
    let b = header("K-fold ordinario con un test final fijo", subtitle) + plot;
    b += foldRows(sc, 206, IID.folds, fold ? k : null);
    b += fold
      ? countsRow([{ role: "train", value: fold.train.length }, { role: "validation", value: fold.validation.length }, { role: "test", value: fold.test.length }], 312)
      : countsRow([{ role: "development", value: IID.development.length }, { role: "test", value: IID.holdout.length }], 312);
    const notes = fold
      ? [`Se entrena con ${fold.train.length} puntos y se valida con ${fold.validation.length} distintos.`, "El test final (rosa, con candado) queda fuera de todos los pliegues."]
      : summary
      ? ["Los 5 grupos de validación cubren los 80 puntos de desarrollo, sin repetir.", "El test final no se usa para elegir hiperparámetros: se mira una vez, al final."]
      : ["Los 80 puntos restantes se reparten al azar en 5 grupos de 16.", "Cada fila P1…P5 es un pliegue; el rosa es el mismo en todas."];
    b += centered(notes, 344, { size: 12, fill: COLORS.ink });
    b += centered(["Línea base IID: supone observaciones intercambiables.", "No es la validación recomendada para pronosticar una serie de tiempo."], 396, { size: 11.5 });
    const label = fold
      ? `Pliegue ${k + 1}/5: train ${fold.train.length} · validación ${fold.validation.length} · test final ${fold.test.length} (fuera del pliegue).`
      : summary
      ? `Los 5 pliegues validaron los ${IID.development.length} puntos de desarrollo una vez cada uno; el test final sigue intacto.`
      : `Test final apartado: ${IID.holdout.length} puntos al azar quedan bloqueados. Desarrollo: ${IID.development.length} puntos.`;
    return { svg: svg("K-fold ordinario: el test final queda fuera de todos los pliegues", b), label };
  }

  // ---------- Lesson 2: why a random split misleads forecasting ----------

  const LEAK_FRAMES = 5;

  function ring(sc, t, color) {
    return `<circle cx="${sc.sx(t)}" cy="${sc.sy(SIGNAL[t].y)}" r="7" fill="none" stroke="${color}" stroke-width="2"/>`;
  }

  // Zoomed window around the chosen pair, so the one-step gap is readable.
  function leakZoom(withValues) {
    const { validation: v, futureTrain: u } = LEAK;
    const roles = foldRoles(IID.folds[LEAKAGE_FOLD]);
    const [t0, t1] = [Math.max(0, v - 5), Math.min(SIGNAL_LENGTH - 1, u + 5)];
    const win = SIGNAL.slice(t0, t1 + 1);
    const ys = win.map((p) => p.y);
    const zs = timeScale({ x: 60, y: 224, width: 290, height: 62, tMin: t0, tMax: t1, yMin: Math.min(...ys) - 0.5, yMax: Math.max(...ys) + 0.5 });
    let b = text(60, 214, `Zoom t = ${t0}…${t1}`, { size: 11.5, weight: "bold", fill: COLORS.muted });
    b += axes(zs, { ticks: [t0, v, u, t1], size: 10, label: "" });
    b += dots(win, zs, { plot: "zoom", set: (p) => roles[p.t], fill: (p) => ROLE_FILL[roles[p.t]], r: 3.6 });
    b += ring(zs, v, ROLE_COLORS.validation) + ring(zs, u, ROLE_COLORS.train);
    const ay = r1(Math.min(zs.sy(SIGNAL[v].y), zs.sy(SIGNAL[u].y)) - 14);
    b += `<g data-role="leak-arrow" data-from="${u}" data-to="${v}">${arrow(zs.sx(u) + 2, ay, zs.sx(v) - 2, ay, COLORS.rose, 2)}</g>`;
    const side = withValues
      ? [
          [`y(${v}) = ${fmt(SIGNAL[v].y)}  validación`, ROLE_COLORS.validation],
          [`y(${u}) = ${fmt(SIGNAL[u].y)}  train`, ROLE_COLORS.train],
          ["Vecinos en el tiempo se parecen:", COLORS.ink],
          ["autocorrelación. El train «sopla»", COLORS.ink],
          ["la respuesta de la validación.", COLORS.ink],
        ]
      : [
          [`t = ${u}: train, en el futuro`, ROLE_COLORS.train],
          [`t = ${v}: validación, en el pasado`, ROLE_COLORS.validation],
          ["Para evaluar t = " + v + " el modelo", COLORS.ink],
          ["ya entrenó con un dato posterior.", COLORS.ink],
          ["Al pronosticar, ese dato no existe.", COLORS.ink],
        ];
    side.forEach(([s, fill], i) => (b += text(372, 232 + i * 18, s, { size: 12, fill, weight: i < 2 ? "bold" : "normal" })));
    return b;
  }

  function renderLeakage(frame) {
    const fold = IID.folds[LEAKAGE_FOLD];
    const roles = foldRoles(fold);
    const { validation: v, futureTrain: u } = LEAK;
    const subtitles = [
      `Pliegue ${LEAKAGE_FOLD + 1} del K-fold aleatorio, sobre el eje de tiempo original`,
      "Train y validación por separado: la misma forma general",
      "Un dato de train está en el futuro de un dato de validación",
      "Puntos vecinos en el tiempo se parecen",
      "Para pronosticar: validar siempre después del train",
    ];
    const { sc, b: plot } = mainPlot(roles);
    let b = header("¿Por qué no barajar al pronosticar?", subtitles[frame]) + plot;
    let notes;

    if (frame === 1) {
      [
        { set: "train", title: `Solo train (${fold.train.length})`, x: 60, idx: fold.train },
        { set: "validation", title: `Solo validación (${fold.validation.length})`, x: 330, idx: fold.validation },
      ].forEach((s) => {
        const ss = timeScale({ x: s.x, y: 222, width: 230, height: 70 });
        b += text(s.x, 214, s.title, { size: 12, weight: "bold", fill: ROLE_FILL[s.set] });
        b += axes(ss, { ticks: [0, 50, 99], size: 10, label: "" });
        b += dots(s.idx.map((t) => SIGNAL[t]), ss, { plot: `${s.set}-only`, set: s.set, fill: ROLE_FILL[s.set], r: 2.4 });
      });
      notes = ["Ambos conjuntos dibujan la misma tendencia y el mismo ciclo.", "Parecerse en la forma no prueba que el modelo pronostique el futuro."];
    } else if (frame === 2 || frame === 3) {
      const [t0, t1] = [Math.max(0, v - 5), Math.min(SIGNAL_LENGTH - 1, u + 5)];
      b += `<rect data-role="zoom-window" x="${r1(sc.sx(t0) - 3)}" y="${sc.y}" width="${r1(sc.sx(t1) - sc.sx(t0) + 6)}" height="${sc.height}" fill="none" stroke="${COLORS.rose}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
      b += ring(sc, v, ROLE_COLORS.validation) + ring(sc, u, ROLE_COLORS.train);
      b += leakZoom(frame === 3);
      notes =
        frame === 2
          ? ["La flecha va del futuro (train) al pasado (validación).", "La validación mira hacia atrás algo que el train ya vio adelante."]
          : ["Con autocorrelación, la validación aleatoria puede resultar optimista.", "La forma parecida tampoco prueba que generalice al futuro."];
    } else if (frame === 4) {
      const chrono = TEMPORAL.expanding.folds[0];
      b += text(60, 216, `Aleatoria (P${LEAKAGE_FOLD + 1}): validación intercalada con train futuro`, { size: 12, weight: "bold", fill: COLORS.ink });
      b += foldRow(sc, 222, roles, { row: "random", fold: LEAKAGE_FOLD });
      b += text(60, 256, `Cronológica: train t = ${span(chrono.train)}, validación t = ${span(chrono.validation)}`, { size: 12, weight: "bold", fill: COLORS.ink });
      b += foldRow(sc, 262, foldRoles(chrono), { row: "chronological", fold: 0 });
      notes = [
        "Si el objetivo es pronosticar, la validación va siempre después del train.",
        "La división aleatoria no siempre está prohibida en datos temporales:",
        "depende de la tarea y de la dependencia entre observaciones.",
      ];
    } else {
      b += centered(["Train (azul) y validación (ámbar) se intercalan a lo largo del tiempo.", "El test final (rosa) sigue fuera, igual que en el K-fold."], 228, { size: 12.5, fill: COLORS.ink });
      notes = ["Cada conjunto toma puntos de todo el eje: pasado y futuro se mezclan.", "La curva gris es la señal original, solo como referencia."];
    }
    b += countsRow([{ role: "train", value: fold.train.length }, { role: "validation", value: fold.validation.length }, { role: "test", value: fold.test.length }], 320);
    b += centered(notes, 346, { size: 12, fill: COLORS.ink });
    b += centered(["Aquí no se ajusta ningún modelo: solo se muestra qué datos ve cada conjunto."], 418, { size: 11.5 });

    const labels = [
      `Pliegue ${LEAKAGE_FOLD + 1} aleatorio: ${fold.train.length} train y ${fold.validation.length} validación mezclados en t = 0…99; test final fuera.`,
      "Solo train y solo validación: los dos conservan la forma general de la señal, pero eso no prueba que se pronostique bien.",
      `Validación en t = ${v}, train en t = ${u}: el modelo vería un dato ${u - v} paso${u - v > 1 ? "s" : ""} posterior al que valida.`,
      `y(${v}) = ${fmt(SIGNAL[v].y)} y y(${u}) = ${fmt(SIGNAL[u].y)}: vecinos parecidos por autocorrelación.`,
      "Para pronosticar, validar después del train; barajar no siempre está prohibido, depende de la tarea.",
    ];
    return { svg: svg("Una división aleatoria mezcla pasado y futuro entre train y validación", b), label: labels[frame] };
  }

  // ---------- Lessons 3 and 4: expanding and sliding temporal validation ----------

  const STAGES = ["fit", "validate", "advance"];
  const TEMPORAL_TICKS = [0, 30, 40, 50, 60, 70, 80, 99];

  function renderTemporal(kind, frame) {
    const data = TEMPORAL[kind];
    const sliding = kind === "sliding";
    const last = 1 + data.folds.length * STAGES.length;
    const k = frame >= 1 && frame < last ? Math.floor((frame - 1) / STAGES.length) : null;
    const stage = k == null ? null : STAGES[(frame - 1) % STAGES.length];
    const fold = k == null ? null : data.folds[k];
    const devRoles = foldRoles({ train: data.development, validation: [], test: data.test }).map((r) => (r === "train" ? "development" : r));
    const roles = fold ? foldRoles(fold) : devRoles;
    const title = sliding ? "Validación temporal: ventana deslizante (ancho fijo)" : "Validación temporal: ventana creciente";
    const subtitle = fold
      ? `Pliegue ${k + 1}/5 · train t = ${span(fold.train)} (${fold.train.length}) · validación t = ${span(fold.validation)} (${fold.validation.length})`
      : `El test final t = ${span(data.test)} queda bloqueado fuera de todos los pliegues`;
    const sc = timeScale(TEMPORAL_BOX);
    let b = header(title, subtitle);
    b += band(sc, data.test[0], data.test[data.test.length - 1], ROLE_COLORS.test, `data-role="test-band"`);
    if (fold) {
      b += band(sc, fold.train[0], fold.train[fold.train.length - 1], ROLE_COLORS.train, `data-role="train-band"`);
      b += band(sc, fold.validation[0], fold.validation[fold.validation.length - 1], ROLE_COLORS.validation, `data-role="validation-band"`);
    }
    b += mainPlot(roles, TEMPORAL_TICKS, TEMPORAL_BOX).b;
    const testMid = r1((sc.sx(data.test[0]) + sc.sx(data.test[data.test.length - 1])) / 2);
    b += lock(testMid - 30, 58) + text(testMid - 20, 66, "test final", { size: 11, fill: ROLE_COLORS.test, weight: "bold" });

    if (fold) {
      const ox = r1((sc.sx(fold.origin - 1) + sc.sx(fold.origin)) / 2);
      b += line(ox, sc.y - 2, ox, sc.y + sc.height, COLORS.ink, 1.5, `data-origin="${fold.origin}" stroke-dasharray="5 4"`);
      b += text(ox, sc.y - 5, `origen t = ${fold.origin}`, { size: 10.5, anchor: "middle", weight: "bold" });
      const vEnd = sc.sx(fold.validation[fold.validation.length - 1]);
      const ay = 68;
      if (stage === "fit") {
        b += text(r1((sc.sx(fold.train[0]) + ox) / 2), ay, "ajustar con el pasado", { size: 12, anchor: "middle", weight: "bold", fill: ROLE_COLORS.train });
      } else if (stage === "validate") {
        b += arrow(ox, ay, vEnd + 2, ay, ROLE_COLORS.validation, 2);
        b += text(r1((ox + vEnd) / 2), ay - 7, "validar", { size: 11.5, anchor: "middle", weight: "bold", fill: ROLE_COLORS.validation });
      } else if (k < data.folds.length - 1) {
        const next = r1((sc.sx(fold.origin + data.step - 1) + sc.sx(fold.origin + data.step)) / 2);
        b += arrow(ox, ay, next, ay, COLORS.ink, 2);
        b += text(r1((ox + next) / 2), ay - 7, "avanza", { size: 11.5, anchor: "middle", weight: "bold" });
      }
    }

    b += foldRows(sc, 204, data.folds, k);
    if (sliding && fold) {
      const cmp = TEMPORAL.expanding.folds[k];
      b += text(60, 300, `Comparación, ventana creciente en P${k + 1}: train t = ${span(cmp.train)} (${cmp.train.length})`, { size: 11, fill: COLORS.muted, weight: "bold" });
      b += foldRow(sc, 304, foldRoles(cmp), { row: "compare", fold: k });
    }

    const items = fold
      ? [
          ...(sliding ? [{ role: "discarded", value: fold.discarded.length }] : []),
          { role: "train", value: fold.train.length },
          { role: "validation", value: fold.validation.length },
          { role: "unused", value: fold.unused.length },
          { role: "test", value: fold.test.length },
        ]
      : [{ role: "development", value: data.development.length }, { role: "test", value: data.test.length }];
    b += countsRow(items, 336);

    const keep = sliding
      ? `La ventana avanza: suma ${data.step} recientes y descarta los ${data.step} más viejos.`
      : `El train crece: conserva todo el pasado y suma ${data.step} puntos.`;
    const notes = !fold
      ? frame === 0
        ? ["Los pliegues solo usan t = 0…79; el test final se mira una vez, al final.", "Cada pliegue: ajustar con el pasado y validar los 10 pasos siguientes."]
        : sliding
        ? ["Variante de ancho fijo (rolling): olvida la historia vieja si el proceso cambia.", "En TimeSeriesSplit equivale a max_train_size=30; por defecto el train crece."]
        : ["Es la idea de TimeSeriesSplit de scikit-learn: train creciente, validación futura.", "Se comparan modelos con los 5 pliegues; el test final no interviene."]
      : stage === "fit"
      ? ["Se ajusta solo con el train del pliegue (también escalado o normalización).", "Gris: futuro que este pliegue todavía no puede usar."]
      : stage === "validate"
      ? [`Se pronostican los ${fold.validation.length} pasos siguientes al origen sin haberlos visto.`, "El train termina antes del origen: máx(train) < mín(validación)."]
      : k < data.folds.length - 1
      ? [keep, sliding ? "Gris claro: historia descartada, distinta del futuro sin usar." : `El origen pasa a t = ${fold.origin + data.step}.`]
      : ["Último pliegue: la validación termina en t = 79.", "El test final t = 80…99 sigue bloqueado."];
    b += centered(notes, 362, { size: 12, fill: COLORS.ink });
    b += centered([`Señal con pasos de tiempo equiespaciados · horizonte fijo de ${data.horizon} pasos.`], 418, { size: 11.5 });

    const trainText = fold ? `train t = ${span(fold.train)} (${fold.train.length})` : "";
    const label = !fold
      ? frame === 0
        ? `Test final t = ${span(data.test)} bloqueado; los pliegues usan solo t = ${span(data.development)}.`
        : sliding
        ? "Cinco pliegues con ventana de ancho fijo 30 y horizonte 10; el test final nunca entró en un pliegue."
        : "Cinco pliegues con train creciente y horizonte 10 (como TimeSeriesSplit); el test final nunca entró en un pliegue."
      : stage === "fit"
      ? `Pliegue ${k + 1}/5: se ajusta con ${trainText}, solo pasado.`
      : stage === "validate"
      ? `Pliegue ${k + 1}/5: se valida en t = ${span(fold.validation)}, después del origen; ${trainText}.`
      : k < data.folds.length - 1
      ? `Pliegue ${k + 1}/5 (${trainText}): el origen avanza a t = ${fold.origin + data.step}.`
      : `Pliegue ${k + 1}/5 (${trainText}): último pliegue; el test t = ${span(data.test)} sigue bloqueado.`;
    return { svg: svg(title, b), label };
  }

  const TEMPORAL_FRAMES = 1 + TEMPORAL.expanding.folds.length * STAGES.length + 1;

  const LESSONS = {
    folds: { frameCount: FOLDS_FRAMES, intervalMs: 1600, title: "K-fold ordinario con test final fijo", render: renderFolds },
    leakage: { frameCount: LEAK_FRAMES, intervalMs: 2600, title: "Por qué barajar no sirve para pronosticar", render: renderLeakage },
    expanding: { frameCount: TEMPORAL_FRAMES, intervalMs: 1300, title: "Validación temporal con ventana creciente", render: (f) => renderTemporal("expanding", f) },
    sliding: { frameCount: TEMPORAL_FRAMES, intervalMs: 1300, title: "Validación temporal con ventana deslizante", render: (f) => renderTemporal("sliding", f) },
  };
  // Registering in the shared registry reuses SpatialLesson's player, controls and keyboard handling.
  Object.keys(LESSONS).forEach((kind) => (S.LESSONS[`validation-${kind}`] = LESSONS[kind]));

  // ---------- DOM mounting ----------

  const instances = new WeakMap();

  function mount(el, kind) {
    if (instances.has(el)) return instances.get(el);
    const key = kind || el.getAttribute("data-validation-lesson");
    if (!Object.prototype.hasOwnProperty.call(LESSONS, key)) throw new Error(`Unknown validation lesson: ${key}`);
    const instance = S.mount(el, `validation-${key}`);
    instance.kind = key;
    instance.root.classList.add("validation-lesson");
    instances.set(el, instance);
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-validation-lesson]"))
      .filter((el) => Object.prototype.hasOwnProperty.call(LESSONS, el.getAttribute("data-validation-lesson")))
      .map((el) => mount(el));
  }

  return {
    IID_DEFAULTS,
    TEMPORAL_DEFAULTS,
    LEAKAGE_FOLD,
    ROLE_FILL,
    LESSONS,
    iidFolds,
    temporalFolds,
    foldRoles,
    leakageExample,
    renderFolds,
    renderLeakage,
    renderTemporal,
    mount,
    autoMount,
  };
});
