# Animación matricial: la convolución como el mismo XW

Widget original en SVG animado, sin dependencias. Reutiliza la casa, los filtros, `windowAt`, `dotTerms`, `convolve2d` y el reproductor de `spatial.js`; todos los números dibujados salen de las mismas funciones que usan las pruebas (`tests/test_matrix.cjs`).

## Cómo montarlo

```html
<link rel="stylesheet" href="animations/spatial.css" />
<link rel="stylesheet" href="animations/matrix.css" />
<div data-matrix-lesson></div>
<script src="animations/spatial.js"></script>
<script src="animations/matrix.js"></script>
```

- `spatial.js` debe cargarse antes: `matrix.js` registra la lección `matrix` en `SpatialLesson.LESSONS` y usa su `mount`, así que los controles son los mismos («Pausar/Reproducir», «Paso siguiente»), igual que el teclado acotado al widget, la pausa al ocultar la diapositiva o la pestaña y el arranque en pausa con `prefers-reduced-motion`.
- El script monta automáticamente cada `[data-matrix-lesson]` y expone `window.MatrixLesson` (`mount`, `autoMount`, `patchMatrix`, `filterMatrix`, `matmul`, `columnToMap`, `matrixData`).

## Qué muestra (40 pasos)

1. **Ventanas → filas de X (9 pasos).** Cada ventana 3×3 de la casa 5×5 (paso 1, sin relleno) se resalta en ámbar y se aplana por filas en una fila de `X_patches` (9×9). Las ventanas se solapan: el píxel central aparece en las 9 filas.
2. **Filtros → columnas de W (3 pasos).** Cada kernel 3×3 se aplana en una **columna** de `W` (9×3): vertical, horizontal y una copia del vertical.
3. **Productos (27 pasos).** Fila `p` de X por columna `f` de W: se ven los nueve productos con sus valores, la celda `Y[p, f]` de `Y` (9×3) y la misma celda en el mapa `f`.
4. **Reacomodo.** Cada columna de Y, leída por filas, es un mapa 3×3. Las columnas 1 y 3 son iguales: el filtro repetido produce el mismo mapa (redundancia).

Resultado, idéntico a la convolución directa con sesgo 0 y sin activación:

| | Mapa 3×3 |
|---|---|
| Filtro 1 (vertical) | `[[2,2,2],[3,1,3],[3,2,3]]` |
| Filtro 2 (horizontal) | `[[1,2,1],[3,3,3],[1,2,1]]` |
| Filtro 3 (copia del 1) | `[[2,2,2],[3,1,3],[3,2,3]]` |

## Idea para el estudiante

Es el mismo `XW` de la capa densa. La diferencia es qué hay en cada fila de X: antes era la imagen entera; ahora es una ventana local, y las ventanas se solapan. W tiene solo 9 pesos por filtro, compartidos por todas las filas. Esta disposición se conoce como im2col (una ventana por fila). Es una forma de entender el cálculo: las bibliotecas pueden obtener el mismo resultado sin copiar las ventanas en memoria. Los kernels son ilustrativos, no entrenados.

## Fuentes (solo referencia)

- CS231n, *Convolutional Neural Networks*, sección «Implementation as Matrix Multiplication»: <https://cs231n.github.io/convolutional-networks/>.

Se usó solo como referencia conceptual; no se copió ningún recurso gráfico de terceros.
