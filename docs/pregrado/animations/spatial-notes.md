# Animaciones espaciales: densa frente a Conv2D, kernel y filtros

Widgets originales en SVG animado, sin dependencias. Todos los números que se dibujan salen de las mismas funciones puras que usan las pruebas (`tests/test_spatial.cjs`).

## Cómo montarlos

```html
<link rel="stylesheet" href="animations/spatial.css" />
<div data-spatial-lesson="comparison"></div>
<div data-spatial-lesson="kernel"></div>
<div data-spatial-lesson="filters"></div>
<script src="animations/spatial.js"></script>
```

- El script monta automáticamente cada `[data-spatial-lesson]` y expone `window.SpatialLesson` (`mount`, `autoMount` y las funciones puras).
- Cada widget crea su SVG (`viewBox` 600×440), una etiqueta de etapa con `aria-live="polite"` y los botones «Pausar/Reproducir» y «Paso siguiente».
- La animación corre solo si la `.slide` que lo contiene tiene `.is-active` (y la pestaña está visible). Al ocultarse la diapositiva se elimina el temporizador.
- Con `prefers-reduced-motion: reduce` arranca en pausa; el paso manual siempre funciona. Un paso manual pausa la reproducción.
- Con el foco dentro del widget, Espacio, flechas, Inicio, Fin, RePág y AvPág no llegan al manejador del deck: flechas avanzan o retroceden un paso, Inicio y Fin van al primer y último paso. Tras un clic con el ratón, el botón suelta el foco para que las flechas vuelvan a mover las diapositivas.

## Qué muestra cada widget

**comparison** — Imagen 7×7 de un canal con la casa 5×5 incrustada. A la izquierda, una capa densa con 25 salidas: cada neurona ve los 49 píxeles con sus propios pesos (el mapa de calor cambia en cada posición). A la derecha, un filtro 3×3 recorre la imagen y usa los mismos 9 pesos en las 25 posiciones, llenando un mapa 5×5. Conteo con las mismas dimensiones de entrada y salida:

| | Por unidad | Capa completa |
|---|---|---|
| Densa (25 neuronas) | 49 + 1 = 50 | 25 × 50 = 1250 |
| Conv2D (1 filtro 3×3) | 9 + 1 = 10 | 10 |

En la segunda mitad la casa se desplaza un píxel: el mismo filtro produce el mapa desplazado una columna. Esto no implica que la clasificación sea invariante, ni que una red densa no pueda aprender imágenes; solo muestra que la convolución reutiliza pesos locales.

**kernel** — Ventana 3×3 sobre la casa 5×5 con el kernel vertical `[[0,1,0],[0,1,0],[0,1,0]]`, paso 1, sin relleno, sesgo 0. Muestra el parche, el kernel, los nueve productos, la suma y el mapa 3×3 que se va llenando: `[[2,2,2],[3,1,3],[3,2,3]]`.

**filters** — La misma casa pasa por tres filtros: vertical, horizontal y una copia del vertical. Los tres mapas se calculan por separado; el tercero coincide con el primero (redundancia). El entrenamiento ajusta los pesos según la tarea y nada obliga a que los filtros sean distintos. Los kernels son ilustrativos, no entrenados.

## Fuentes (solo referencia)

- CS231n, *Convolutional Neural Networks*: <https://cs231n.github.io/convolutional-networks/> — campos receptivos locales, pesos compartidos y conteo de parámetros.
- vdumoulin/conv_arithmetic: <https://github.com/vdumoulin/conv_arithmetic> — animaciones de la ventana deslizante (por ejemplo `gif/no_padding_no_strides.gif`).

Ambas se usaron solo como referencia conceptual. No se copió ningún GIF ni recurso gráfico de terceros; las figuras se generan en el navegador a partir de código propio.
