# Teoría de Aprendizaje de Máquina

Material del curso: notebooks, prácticas y presentaciones.

**Diapositivas del curso:**

- Posgrado: https://daprosero.github.io/Teoria-Aprendizaje-de-Maquina/
- Pregrado: https://daprosero.github.io/Teoria-Aprendizaje-de-Maquina/pregrado/

## Estructura

```
.
├── pregrado/   # material del curso de pregrado
└── posgrado/   # material del curso de posgrado
```

### posgrado

| Clase | Contenido |
|-------|-----------|
| Clase 1 | Procesamiento de datos (teoría y práctica) |
| Clase 2 | Regresión, clasificación, conglomerados, reducción de dimensión, búsqueda de hiperparámetros |
| Clase 3 | Redes neuronales artificiales y aprendizaje profundo |
| Clase 4 | Series de tiempo, detección de objetos, aprendizaje por refuerzo y agentes inteligentes |

En la raíz de `posgrado/` están además la introducción a ciencia de datos e IA y notebooks de apoyo.

### pregrado

Presentación publicada: https://daprosero.github.io/Teoria-Aprendizaje-de-Maquina/pregrado/

Las diapositivas están en [`docs/pregrado/`](docs/pregrado/index.html) (56 diapositivas):
reducción de dimensión (PCA, Kernel PCA, t-SNE, UMAP), conglomerados, panorama de
regresión y clasificación con scikit-learn, redes neuronales y redes convolucionales.

El cierre de redes convolucionales tiene lecciones animadas propias (SVG + JavaScript,
sin dependencias):

| Diapositiva | Animación |
|-------------|-----------|
| 53 | Capa densa: entradas, pesos, suma y ReLU |
| 54 | Densa frente a Conv2D: 1250 frente a 10 parámetros para 49 entradas y 25 salidas |
| 55 | Un kernel 3 × 3 recorre la imagen y llena el mapa de características |
| 56 | Banco de filtros: vertical, horizontal y uno redundante |
| 57 | La convolución como el mismo `XW`: ventanas → filas de `X`, filtros → columnas de `W` |
| 58 | Paso (stride): kernel 3 × 3 sobre 5 × 5 con paso 1 (mapa 3 × 3) y paso 2 (mapa 2 × 2) |
| 59 | Relleno (padding): sin relleno el mapa se achica a 3 × 3; con relleno 1 de ceros conserva 5 × 5 |
| 60 | Max pooling 2 × 2, paso 2: de 4 × 4 a 2 × 2 conservando el mayor, sin pesos |
| 61 | Average pooling sobre la misma entrada, comparado con max pooling |
| 62 | Dropout: máscaras distintas en entrenamiento (p = 0.5, reescala por 0.5); en inferencia no se apaga nada |
| 63 | Recorrido completo: 6 × 6 → conv 4 × 4 × 2 → ReLU → pooling 2 × 2 × 2 → 8 → dos puntajes |

Controles: «Pausar/Reproducir» y «Paso siguiente». Con el foco en la animación, las
flechas avanzan o retroceden un paso; fuera de ella, mueven las diapositivas. Con
`prefers-reduced-motion` arranca en pausa.

**Publicación.** `docs/` es la fuente del sitio. El flujo
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) lo despliega en GitHub Pages
en cada push a `main` que toque `docs/**`; la versión publicada refleja lo que ya está
en `main`, así que las diapositivas nuevas aparecen allí solo después de integrarlas.

**Ver en local:**

```bash
python3 -m http.server 8011 --bind 127.0.0.1 --directory docs
# abrir http://127.0.0.1:8011/pregrado/
```

**Pruebas:**

```bash
node --test tests/test_*.cjs
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

**Referencias.** [CS231n · Convolutional Networks](https://cs231n.github.io/convolutional-networks/),
[vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) y, como
lectura opcional, [CNN Explainer](https://poloclub.github.io/cnn-explainer/). Se usaron solo
como referencia conceptual: las animaciones son originales y no incluyen GIF ni
recursos gráficos de terceros.

## Uso

Los cuadernos están escritos para **Google Colab** y siguen funcionando allí sin cambios.

Para ejecutarlos **en local**, cada nivel trae su propio entorno:

```bash
cd posgrado
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El detalle está en [`posgrado/README.md`](posgrado/README.md): kernel de Jupyter,
dónde quedan los datos descargados y qué cuadernos tienen limitaciones fuera de Colab.

> Los datos que los cuadernos clonan o descargan (repositorios, Google Drive,
> Kaggle, pesos de modelos) van a `posgrado/.cache/` y **no se suben** al
> repositorio.
