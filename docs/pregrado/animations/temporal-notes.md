# Animaciones de series de tiempo: señal, autorregresión y división train/test

Widgets originales en SVG animado, sin dependencias. Reutilizan el reproductor, los controles, el teclado y los colores de `spatial.js`; todos los números dibujados salen de las mismas funciones que usan las pruebas (`tests/test_temporal.cjs`).

## Cómo montarlos

```html
<link rel="stylesheet" href="animations/spatial.css" />
<link rel="stylesheet" href="animations/temporal.css" />
<div data-temporal-lesson="signal"></div>
<div data-temporal-lesson="autoregression"></div>
<div data-temporal-lesson="split"></div>
<script src="animations/spatial.js"></script>
<script src="animations/temporal.js"></script>
```

- `spatial.js` debe cargarse antes: `temporal.js` registra `temporal-signal`, `temporal-autoregression` y `temporal-split` en `SpatialLesson.LESSONS` y usa su `mount`. Los controles («Pausar/Reproducir», «Paso siguiente»), el teclado acotado al widget, la pausa al ocultar la diapositiva o la pestaña y el arranque en pausa con `prefers-reduced-motion` son los mismos.
- El script monta automáticamente cada `[data-temporal-lesson]` con un tipo conocido y expone `window.TemporalLesson`. `mount(el, tipo)` es idempotente y devuelve la instancia de `SpatialLesson.mount` (`root`, `stage`, `label`, `playButton`, `nextButton`, `player`, `destroy`) más `kind`.
- La división además devuelve `options`, `modeButtons.random`, `modeButtons.chronological`, `fractionInput` (rango 10–40, paso 10), `fractionValue`, `seedButton`, `getConfig()`, `getSplit()` y `setConfig({ mode, testFraction, seed })`. Cada widget tiene su propio estado. Un cambio valida con `partition`, redibuja la asignación completa y pausa como un paso manual. Las flechas sobre el control deslizante mueven el control, no la animación ni el mazo.

## Datos y funciones puras

- `SIGNAL`: 100 puntos congelados `{ t, y }`, con `t = 0..99` y `y` redondeado a dos decimales. `signalValues()` devuelve una copia `values[t] = y(t)`. Es una señal **sintética**: `y(t) = 10 + 0,04·t + 2·sen(2πt/20) + ruido`, ruido uniforme en ±0,6 con semilla 7 (`generateSignal(semilla)`). No fue generada por el modelo autorregresivo.
- `mulberry32(semilla)` y `seededShuffle(lista, semilla)`: aleatoriedad reproducible; no modifican la entrada.
- `partition({ mode: "random" | "chronological", testFraction = 0.2, seed = 7 })` → `{ mode, testFraction, seed, train, test }` con índices de tiempo ordenados. Aleatoria: barajado con semilla, `round(100·fracción)` puntos de test en cualquier lugar del eje. Cronológica: el test es el último bloque contiguo (por defecto `t = 80..99`) y la semilla no interviene (`seed: null`). Siempre disjuntos y juntos cubren las 100 observaciones. El test no se usa para ajustar nada.
- `arForecast(valores, t)` → `{ target, lags, prediction, observed, error }` con `ŷ(t) = 0,6·y(t−1) + 0,25·y(t−2) + 0,1·y(t−3)` (`AR_COEFFS`, rezago más reciente primero). Solo lee valores anteriores a `t`; `t` debe estar entre 3 y 99. `lagTable(valores)` arma una fila cronológica por objetivo. `contiguousRuns(índices)` separa tramos sin huecos.
- `plot`: primitivas SVG para las lecciones de validación (`timeScale`, `axes`, `seriesPath`, `dots`, `band`, `diamond`, `arrow`, `text`, `line`, `svg`, `centered`). `seriesPath` nunca une puntos a través de observaciones faltantes; `dots` marca cada punto con `data-plot`, `data-set`, `data-t` y `data-y`. Colores por rol en `ROLE_COLORS`: train azul, validación ámbar, test final rosa.

## Qué muestra cada una

1. **Señal en el tiempo (23 pasos).** Primero la casa 5×5: la vecindad de una imagen es espacial. Después aparece un eje temporal y los puntos van llegando de a cinco, con un reloj y el valor del instante actual. Al final se marcan la tendencia y un ciclo de unos 20 pasos.
2. **Autorregresión (24 pasos).** El origen del pronóstico avanza de `t = 30` a `t = 37`. En cada origen: los tres rezagos (entradas, en azul) con sus pesos apuntan a un «?»; se calcula `ŷ(t)` con números reales; recién entonces se revela `y(t)` y el error. Lo que está después del origen no se dibuja antes de la revelación. Una fila cronológica separa rezagos y objetivo.
3. **División train/test (12 pasos, interactiva).** Curva original tenue de referencia y puntos de train (azul) y test (rosa) sobre el eje de tiempo original; la asignación se revela de izquierda a derecha. Debajo, train solo y test solo, con puntos sin líneas. Los botones cambian entre aleatoria (IID) y cronológica, el control deslizante cambia el porcentaje de test y «Nueva semilla» sortea otra partición aleatoria (desactivado en modo cronológico).

| Señal en `t` | 28 | 29 | 30 | ŷ(31) | y(31) |
|---|---|---|---|---|---|
| valores | 11.78 | 12.27 | 11.26 | 11.00 | 10.77 |

## Idea para el estudiante

- En una serie de tiempo el orden es parte del dato: el valor de hoy depende de los anteriores.
- Un modelo autorregresivo usa como entradas sus propios valores pasados. Los coeficientes son ilustrativos, no entrenados: el error mostrado no mide la calidad de ningún ajuste.
- Con una división aleatoria, train y test cubren todo el eje y suelen conservar la misma forma general. Es una línea base IID, no una prueba de que el modelo pronostique el futuro. La división cronológica deja el test después de todo el train, como al pronosticar.
- La división es una demostración retrospectiva sobre datos ya observados; el test nunca es entrada del modelo.

## Fuentes (solo referencia)

- scikit-learn, `TimeSeriesSplit`: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>.
- Hyndman y Athanasopoulos, *Forecasting: Principles and Practice* (3.ª ed.), validación cruzada en series de tiempo: <https://otexts.com/fpp3/tscv.html>.

Se usaron solo como referencia conceptual; no se copió ningún recurso gráfico de terceros.
