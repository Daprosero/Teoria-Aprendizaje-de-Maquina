# Animaciones de validación: K-fold, fuga temporal, ventana creciente y ventana deslizante

Widgets originales en SVG animado, sin dependencias. Reutilizan el reproductor, los controles, el teclado y los colores de `spatial.js`, y la señal canónica, `partition` y las primitivas de dibujo de `temporal.js`. Todos los índices y conteos dibujados salen de las mismas funciones que usan las pruebas (`tests/test_validation.cjs`).

## Cómo montarlos

```html
<link rel="stylesheet" href="animations/spatial.css" />
<link rel="stylesheet" href="animations/temporal.css" />
<link rel="stylesheet" href="animations/validation.css" />
<div data-validation-lesson="folds"></div>
<div data-validation-lesson="leakage"></div>
<div data-validation-lesson="expanding"></div>
<div data-validation-lesson="sliding"></div>
<script src="animations/spatial.js"></script>
<script src="animations/temporal.js"></script>
<script src="animations/validation.js"></script>
```

- Orden obligatorio: `spatial.js`, `temporal.js`, `validation.js`. Se registran `validation-folds`, `validation-leakage`, `validation-expanding` y `validation-sliding` en `SpatialLesson.LESSONS` y se montan con `SpatialLesson.mount`: mismos controles («Pausar/Reproducir», «Paso siguiente»), teclado acotado al widget, pausa al ocultar la diapositiva o la pestaña y arranque en pausa con `prefers-reduced-motion`. No agrega temporizadores propios.
- El script monta automáticamente cada `[data-validation-lesson]` con un tipo conocido y expone `window.ValidationLesson`. `mount(el, tipo)` es idempotente y devuelve la instancia de `SpatialLesson.mount` (`root`, `stage`, `label`, `playButton`, `nextButton`, `player`, `destroy`) más `kind`.

## Datos y funciones puras

- `iidFolds({ k = 5, seed = 7 })` → `{ holdout, development, folds }`. El test final se aparta **una sola vez** con `partition({ mode: "random", testFraction: 0.2, seed: 7 })` (20 índices al azar). Los 80 de desarrollo se barajan con `seed` y se cortan en `k` grupos de validación; cada pliegue es `{ index, train, validation, test }` con 64 / 16 / 20. El `test` es idéntico en todos los pliegues y nunca entra en train ni validación; cada punto de desarrollo es validación exactamente una vez.
- `temporalFolds({ kind: "expanding" | "sliding", initialTrain = 30, horizon = 10, step = 10, folds = 5, trainWidth })` → `{ kind, development, test, folds, … }`. El test final es el último bloque cronológico `t = 80..99`. El origen del pliegue `i` es `30 + 10·i`; la validación son los 10 pasos siguientes.
  - Creciente: train `0..29`, `0..39`, …, `0..69`; conserva todo el pasado y crece de a 10.
  - Deslizante: train de ancho fijo 30, `0..29`, `10..39`, …, `40..69`; la historia más vieja queda en `discarded`.
  - Cada pliegue trae `origin`, `discarded`, `train`, `validation`, `unused` (desarrollo futuro que el pliegue aún no puede usar) y `test`. Siempre `máx(train) < mín(validación)` y todo train y validación `< 80`. Parámetros inválidos, o pliegues que llegarían al test final, lanzan `RangeError`.
- `foldRoles(pliegue)` → un rol por instante (`discarded`, `train`, `validation`, `unused`, `test`). `leakageExample()` → `{ validation, futureTrain, gap }`: en el pliegue 1 aleatorio, un instante de validación con un instante de train 1–3 pasos después (hoy `t = 26` y `t = 27`), nunca del test final.
- Todas devuelven copias nuevas y no modifican la señal ni las opciones recibidas.

## Qué muestra cada una

1. **K-fold ordinario (7 pasos).** Se aparta el test final (rosa, con candado). Después los 5 pliegues: la validación ámbar se mueve y el train azul es su complemento; las filas P1…P5 están siempre visibles. Al final, cada punto de desarrollo fue validación una vez. Se presenta como **línea base IID**, no como la validación recomendada para pronosticar.
2. **Por qué no barajar al pronosticar (5 pasos).** El mismo pliegue aleatorio sobre el eje original: train y validación por separado conservan la forma general, pero eso no prueba que se pronostique bien. Un zoom marca una validación en `t = 26` y un train en `t = 27` con una flecha del futuro al pasado y sus valores reales. Cierra comparando la fila aleatoria con una cronológica. No se ajusta ningún modelo ni se inventan métricas.
3. **Ventana creciente (17 pasos).** Para cada pliegue: ajustar con el pasado, validar los 10 pasos siguientes y avanzar el origen. Gris: desarrollo futuro sin usar; rosa con candado: test final.
4. **Ventana deslizante (17 pasos).** Igual, con train de ancho fijo 30: la historia descartada (gris claro) se distingue del futuro sin usar (gris). Una fila extra compara con la ventana creciente del mismo pliegue.

## Idea para el estudiante

- El test final se aparta antes de todo y no se usa para elegir modelos ni hiperparámetros; se mira una vez, al final.
- Cualquier preprocesamiento (escalado, normalización, selección de variables) se ajusta solo con el train de cada pliegue.
- En pronóstico con autocorrelación, una división aleatoria deja que el train vea el futuro inmediato de la validación. No es que barajar esté siempre prohibido en datos temporales: depende de la tarea y de la dependencia entre observaciones.
- Se supone una señal con pasos de tiempo equiespaciados y un horizonte fijo de 10 pasos.
- La ventana creciente es el comportamiento por defecto de `TimeSeriesSplit` (con `n_splits=5, test_size=10` sobre `t = 0..79` da exactamente estos pliegues). La ventana deslizante de ancho fijo es una variante (`max_train_size=30`), no el valor por defecto.

## Fuentes (solo referencia)

- scikit-learn, `TimeSeriesSplit`: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>.
- Hyndman y Athanasopoulos, *Forecasting: Principles and Practice* (3.ª ed.), validación cruzada en series de tiempo: <https://otexts.com/fpp3/tscv.html>.

Se usaron solo como referencia conceptual; no se copió ningún recurso gráfico de terceros.
