# Animaciones de dropout y recorrido completo de una CNN

Widgets originales en SVG animado, sin dependencias. Reutilizan la casa 5×5, los kernels y el reproductor de `spatial.js`, y `convolve` y `pool2d` de `operations.js`; todos los números dibujados salen de las mismas funciones que usan las pruebas (`tests/test_regularization.cjs`).

## Cómo montarlos

```html
<link rel="stylesheet" href="animations/spatial.css" />
<link rel="stylesheet" href="animations/regularization.css" />
<div data-regularization-lesson="dropout"></div>
<div data-regularization-lesson="pipeline"></div>
<script src="animations/spatial.js"></script>
<script src="animations/operations.js"></script>
<script src="animations/regularization.js"></script>
```

- `spatial.js` y `operations.js` deben cargarse antes: `regularization.js` registra las lecciones `cnn-dropout` y `cnn-pipeline` en `SpatialLesson.LESSONS` y usa su `mount`. Los controles («Pausar/Reproducir», «Paso siguiente»), el teclado acotado al widget, la pausa al ocultar la diapositiva o la pestaña y el arranque en pausa con `prefers-reduced-motion` son los mismos. Cada widget tiene su propio estado.
- El script monta automáticamente cada `[data-regularization-lesson]` con un tipo conocido y expone `window.RegularizationLesson`.
- Función pura: `dropout(activaciones, { rate = 0.5, mask, training = true })`, dropout invertido. En entrenamiento, los valores con máscara 1 se dividen por `1 − rate` y los de máscara 0 pasan a 0; la máscara es explícita (0 o 1, misma forma) para que la lección sea determinista. En inferencia devuelve una copia idéntica, ignora la máscara y no reescala. `rate` debe estar en `[0, 1)`; con `rate = 0` no se apaga nada. Acepta vectores o grillas de números finitos, conserva la forma y no modifica sus entradas.
- Datos de las figuras: `dropoutData()` (activaciones, máscaras, salidas por pasada, inferencia) y `pipelineData()` (lienzo, filtros, mapas de convolución, ReLU, pooling, vector aplanado, pesos de la densa, puntajes y formas).

## Qué muestra cada una

1. **Dropout (10 pasos).** Seis activaciones `[2, 1, 3, 2, 4, 1]` llegan a la capa siguiente. Después, cuatro pasadas de entrenamiento con `p = 0.5`, cada una con una máscara distinta: primero se tachan las activaciones apagadas, luego las que siguen se dividen por 0.5 (valen el doble). La primera activación vale 4 cuando sobrevive y 0 cuando se apaga. A la derecha se acumulan las máscaras usadas. Al final, inferencia: sin máscara y sin reescalar, `2 → 2`. Siempre son 6 valores: no achica como pooling.
2. **Recorrido completo (7 pasos).** La casa en un lienzo 6×6 (una fila y una columna de ceros extra, para que los mapas 4×4 se agrupen justo de a 2×2) → dos filtros 3×3 → ReLU → max pooling 2×2 → aplanar → densa 8×2 → dos puntajes. Cada paso resalta su bloque y un valor con su origen; los mapas anteriores quedan visibles. El último paso muestra la cadena de formas y dónde iría dropout (solo en entrenamiento; aquí no se aplica).

| Paso | Forma | Valores |
|---|---|---|
| Imagen | 6×6 | casa + fila y columna de ceros |
| Convolución (vertical; contraste `[[0,0,0],[1,1,1],[-1,-1,-1]]`) | 4×4×2 | `[[2,2,2,1],[3,1,3,1],[3,2,3,1],[2,1,2,0]]`; `[[-2,-1,-2,-1],[2,1,2,1],[-1,-1,-1,0],[2,3,2,1]]` |
| ReLU | 4×4×2 | los 7 negativos del mapa 2 pasan a 0 |
| Max pooling 2×2, paso 2 | 2×2×2 | `[[3,3],[3,3]]`; `[[2,2],[3,2]]` |
| Aplanar (mapa 1 por filas, después mapa 2) | 8 | `3, 3, 3, 3, 2, 2, 3, 2` |
| Densa 8×2, sesgo 0 | 2 | clase 1 = 6, clase 2 = 3 (puntajes crudos, sin softmax) |

## Idea para el estudiante

- Dropout apaga activaciones distintas en cada pasada de entrenamiento; cada una se apaga con probabilidad `p`, así que cuántas se apagan varía. No elimina neuronas: la que se apaga vuelve en otra pasada. En inferencia no se apaga nada.
- Regulariza: reduce la co-adaptación entre neuronas. No garantiza mejores resultados.
- Las máscaras son fijas e ilustrativas; en el entrenamiento real se sortean al azar.
- En el recorrido, los filtros y los pesos de la densa son parámetros ilustrativos, no entrenados: los puntajes salen de esas cuentas, no de un modelo que clasifique de verdad. El orden del aplanado es una convención (otras bibliotecas intercalan los canales); lo importante es usar siempre el mismo.

## Fuentes (solo referencia)

- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR, 2014): <https://jmlr.org/papers/v15/srivastava14a.html>.
- CS231n, *Neural Networks Part 2* (dropout invertido): <https://cs231n.github.io/neural-networks-2/>.

Se usaron solo como referencia conceptual; no se copió ningún recurso gráfico de terceros.
