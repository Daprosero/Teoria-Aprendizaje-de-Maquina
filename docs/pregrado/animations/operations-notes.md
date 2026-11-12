# Animaciones de geometría CNN: paso, relleno, max pooling y average pooling

Widgets originales en SVG animado, sin dependencias. Reutilizan la casa 5×5, el kernel vertical, `windowAt`, `dotTerms` y el reproductor de `spatial.js`; todos los números dibujados salen de las mismas funciones que usan las pruebas (`tests/test_operations.cjs`).

## Cómo montarlos

```html
<link rel="stylesheet" href="animations/spatial.css" />
<link rel="stylesheet" href="animations/operations.css" />
<div data-operation-lesson="stride"></div>
<div data-operation-lesson="padding"></div>
<div data-operation-lesson="maxpool"></div>
<div data-operation-lesson="averagepool"></div>
<script src="animations/spatial.js"></script>
<script src="animations/operations.js"></script>
```

- `spatial.js` debe cargarse antes: `operations.js` registra las lecciones `cnn-stride`, `cnn-padding`, `cnn-maxpool` y `cnn-averagepool` en `SpatialLesson.LESSONS` y usa su `mount`. Los controles («Pausar/Reproducir», «Paso siguiente»), el teclado acotado al widget, la pausa al ocultar la diapositiva o la pestaña y el arranque en pausa con `prefers-reduced-motion` son los mismos. Cada widget tiene su propio estado.
- El script monta automáticamente cada `[data-operation-lesson]` con un tipo conocido y expone `window.OperationLesson`.
- Funciones puras reutilizables: `convolve(imagen, kernel, { stride = 1, padding = 0 })` (sin voltear el kernel, como en deep learning), `padZeros(imagen, p)`, `pool2d(imagen, { size = 2, stride = 2, mode = "max" | "average" })`, `poolWindow(ventana, modo)`, `outputSize(n, k, paso, relleno)` y `windowPositions(n, k, paso, relleno)`. Aceptan solo matrices cuadradas de números finitos, no modifican sus entradas y rechazan pasos, rellenos o ventanas no válidos.

## Qué muestra cada una

1. **Paso (10 pasos).** La misma casa y el mismo kernel con paso 1 y paso 2, sincronizados: en cada paso la ventana de arriba avanza 1 píxel y la de abajo 2 (la posición anterior queda punteada). El valor calculado viaja a su celda. Paso 1 llena un mapa 3×3 en 9 posiciones; paso 2 llena un mapa 2×2 en 4 y termina antes. Tamaño: `(5 − 3) / paso + 1`.
2. **Relleno (35 pasos).** Primero, sin relleno: 9 posiciones, mapa 3×3. Después, un borde de ceros (en ocre, distinto de los píxeles reales blancos y negros): 25 posiciones, mapa 5×5. La ventana muestra cuántos ceros de relleno incluye. Al final se marca que el centro 3×3 del mapa con relleno es exactamente el mapa sin relleno: los ceros no agregan información de la imagen.
3. **Max pooling (9 pasos).** Ventana 2×2, paso 2, sin relleno, sobre un mapa de activación 4×4 ilustrativo. Cada ventana muestra sus cuatro valores, marca el mayor (en un empate se marcan todos los iguales) y lo lleva a la salida 2×2.
4. **Average pooling (9 pasos).** La misma entrada y la misma ventana, pero `suma / 4`. El mapa de max pooling queda al lado, en el mismo cuadro, para comparar región por región.

| Entrada 4×4 (`POOL_INPUT`) | Max pooling | Average pooling |
|---|---|---|
| `[[1,3,0,2],[2,9,1,1],[0,1,4,4],[2,3,4,4]]` | `[[9,2],[3,4]]` | `[[3.75,1],[1.5,4]]` |

## Idea para el estudiante

- El paso decide cuánto salta la ventana: con paso 2 se visitan menos posiciones y la salida es más chica.
- El relleno de ceros permite centrar la ventana sobre los píxeles del borde y conservar el tamaño; no agrega información real.
- Pooling no tiene pesos que aprender. Max conserva el pico (9); el promedio lo diluye (3.75). En una región pareja dan lo mismo (4). Da cierta tolerancia a corrimientos pequeños dentro de la ventana, pero no garantiza invariancia.
- La casa, el kernel y el mapa de activación son ilustrativos, no entrenados.

## Fuentes (solo referencia)

- Dumoulin y Visin, *A guide to convolution arithmetic for deep learning* (2016): <https://arxiv.org/abs/1603.07285>.
- CS231n, *Convolutional Neural Networks*: <https://cs231n.github.io/convolutional-networks/>.

Se usaron solo como referencia conceptual; no se copió ningún recurso gráfico de terceros.
