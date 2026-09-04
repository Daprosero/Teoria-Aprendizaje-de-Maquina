# Posgrado — Teoría de Aprendizaje de Máquina

## 📊 Diapositivas del curso

**https://daprosero.github.io/Teoria-Aprendizaje-de-Maquina/**

Seis módulos, 62 diapositivas, con la teoría y los ejemplos de estas clases:

| Módulo | Contenido |
|--------|-----------|
| 1. Datos: de crudos a utilizables | `.info()` y `.describe()` como diagnóstico, mecanismos de datos faltantes, las cinco estrategias de imputación, fuga de información y escalado |
| 2. Gráficas y estadística descriptiva | Histograma, boxplot, violín, dispersión, barras, ECDF y matriz de correlación: qué pregunta responde cada uno y cómo se interpreta |
| 3. Reducción de dimensión | Maldición de la dimensión, PCA, cuántas componentes conservar, kernel PCA, t-SNE y UMAP |
| 4. Conglomerados | Aprender sin etiquetas, K-means y su supuesto oculto, cómo elegir k, DBSCAN, espectral y Mini-Batch, validación de grupos |
| 5. Regresión | Esperanza condicional, mínimos cuadrados, sesgo-varianza, Ridge/Lasso/ElasticNet, árboles y KNN, el truco del kernel, SVR, Kernel Ridge, procesos gaussianos, métricas y validación cruzada |
| 6. Probabilidad y clasificación | Probabilidad condicional y Bayes, regresión logística, Naive Bayes, k-vecinos, árboles, bosques, boosting, SVM, matriz de confusión y umbral por costo |

El orden va de lo no supervisado a lo supervisado, para que la matemática se
cobre dos veces: la inercia que minimiza K-means es la misma que minimiza un
árbol de regresión, y el kernel de PCA es el de SVR y los procesos gaussianos.

Cada modelo se presenta con tres bloques fijos —la matemática, una analogía y su
estimador de scikit-learn— y la probabilidad se explica sobre un ejemplo de dos
cajas con bolitas verdes y rojas que después sostiene toda la evaluación.

---

Cuadernos de las clases 1 a 4. Están escritos para Google Colab, pero también
se ejecutan en local con el entorno que se describe abajo.

## Entorno local

Requiere Python 3.11.

```bash
cd posgrado
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name curso-posgrado \
    --display-name "Python 3.11 (curso posgrado)"
```

Luego abrir JupyterLab y elegir el kernel **Python 3.11 (curso posgrado)**:

```bash
jupyter lab
```

## Dónde quedan los datos descargados

Los cuadernos clonan repositorios de GitHub y bajan archivos de Google Drive y
Kaggle. Nada de eso entra al repositorio: todo va a `posgrado/.cache/`, que está
en `.gitignore`.

```
posgrado/.cache/
├── content/       # directorio de trabajo (equivale a /content en Colab)
│   └── curso_IA_CHEC/   # repositorio del curso, clonado una sola vez
├── pip/           # caché de paquetes
├── kagglehub/     # datasets de Kaggle
├── hf/            # modelos de Hugging Face
├── torch/         # pesos de PyTorch
└── ultralytics/   # configuración de YOLO
```

La ubicación se puede cambiar con la variable de entorno `CURSO_CACHE`:

```bash
export CURSO_CACHE=/ruta/a/mi/cache
```

Cada cuaderno arranca con una celda que llama a `curso_setup.init()`. Esa celda
**no altera la ejecución en Colab**: allí detecta el entorno y no hace nada, de
modo que los cuadernos siguen funcionando igual para los estudiantes.

## Verificar que todo corre

```bash
python scripts/run_notebooks.py              # todos los cuadernos
python scripts/run_notebooks.py Regresion    # solo los que coincidan
```

Escribe el detalle en `.cache/informe_ejecucion.json` y las copias ejecutadas en
`.cache/ejecutados/`.

## GPU: el mismo cuaderno en Windows, Linux y macOS

Los cuadernos que entrenan modelos ya no fijan `device=0`. Llaman a
`curso_setup.dispositivo()`, que elige el acelerador de la máquina:

| Devuelve | Cuándo | Plataforma |
|----------|--------|------------|
| `0` | Hay GPU NVIDIA con CUDA | Windows, Linux, Colab con GPU |
| `"mps"` | Hay GPU de Apple Silicon (Metal) | macOS M1/M2/M3/M4 |
| `"cpu"` | No hay acelerador | cualquiera |

El formato es el que esperan Ultralytics/YOLO y PyTorch, así que el mismo
cuaderno corre sin editar nada en las tres plataformas. Para comprobar qué
detecta tu máquina:

```python
import curso_setup
curso_setup.dispositivo()        # 0, "mps" o "cpu"
curso_setup.dispositivo_torch()  # el torch.device equivalente
```

En macOS se activa además `PYTORCH_ENABLE_MPS_FALLBACK=1`, para que las
operaciones que Metal todavía no implementa caigan a CPU en vez de fallar.

**TensorFlow en macOS**: por omisión usa CPU. Para aprovechar la GPU de Apple
se puede instalar `tensorflow-metal`, que no viene en `requirements.txt` porque
en algunas versiones altera resultados numéricos:

```bash
pip install tensorflow-metal
```

## Estado de ejecución

Resultado del barrido con `scripts/run_notebooks.py` sobre macOS (Apple Silicon,
CPU/MPS). Los cuadernos que fallan lo hacen por razones concretas, no por el
entorno.

### Pasan sin errores (10 de 19)

`Clase 1/Procesamiento de datos` · `Clase 1/Práctica - procesamiento de datos` ·
`Clase 2/Clasificación` · `Clase 2/Regresion` · `Clase 3/Práctica_ANN` ·
`Clase 4/Series` · `Clase 4/rl_dummy_colab` · `Copia de Clasificación` ·
`Copia de Regresion` · `Copia de Streamlit`

> Con 8 GB de RAM conviene ejecutarlos por tandas. Los cuadernos con TensorFlow
> no liberan memoria entre kernels y el sistema empieza a paginar: uno que tarda
> 15 s aislado puede tardar 10 min o morir si corre después de varios otros.

### No pueden pasar sin interacción humana (3)

| Cuaderno | Celda | Causa |
|----------|-------|-------|
| `Clase 2/Conglomerados` | 20 | `self.kmeans` se asigna solo dentro del callback de un widget |
| `Clase 2/Reduccion Dimension` | 8 | `visualizer.X` se asigna solo dentro del callback de un widget |
| `Clase 3/Aprendizaje_profundo` | 39 | `history` se asigna solo al pulsar el botón de entrenamiento |

Son **interactivos por diseño**: la clase inicializa el atributo en `None` y solo
lo llena cuando alguien mueve un control. Ninguna librería arregla esto; en una
clase en vivo funcionan bien.

### Requieren GPU NVIDIA (2)

`Clase 2/Conglomerados y Reduccion de Dimenciones Práctica` y
`Práctica_Conglomerados-Reducción` importan `cuml` (RAPIDS), que solo existe para
Linux con CUDA. No hay equivalente en macOS.

### Requieren credenciales (3)

| Cuaderno | Falta |
|----------|-------|
| `Clase 4/2_Detección_de_Objetos` | `ROBOFLOW_API_KEY` (y `WANDB_API_KEY`) |
| `Clase 4/Agentes_Inteligentes` | `OPENAI_API_KEY` |
| `Clase 4/Untitled0` | `OPENAI_API_KEY` |

La lógica de estos cuadernos está verificada: ejecutados con una clave ficticia
llegan hasta la llamada a la API y fallan con un `401`, que es lo esperado.
Comprobalo con `python scripts/check_logica_agentes.py`.

En `2_Detección_de_Objetos` el dispositivo ya se elige solo. En Apple Silicon
selecciona MPS, pero el **entrenamiento** de YOLO sobre Metal choca con un error
conocido de PyTorch; la inferencia sí funciona. Para entrenar, usá CUDA o forzá
`device="cpu"`.

### Con un defecto de contenido (1)

`Clase 2/Búsqueda de hiperparámetros`, celda 25: aplica `QuadraticDiscriminantAnalysis`
sobre MNIST reducido a 200 muestras. Con 10 clases y 784 características quedan
~16 muestras por clase, y la matriz de covarianza no alcanza rango completo.
scikit-learn ≥ 1.6 lanza `LinAlgError`; las versiones anteriores solo advertían.
Fijar scikit-learn a una versión vieja no es opción porque `umap-learn` exige ≥ 1.6.

Sale de tres formas, y es una decisión pedagógica:

1. subir `sample_size` a ~15 000 para que haya más muestras por clase que características;
2. reducir dimensión (PCA) antes del QDA;
3. quitar el QDA de esa sección.

## Scripts

| Script | Para qué sirve |
|--------|----------------|
| `curso_setup.py` | Prepara el directorio de trabajo y las cachés. Lo llaman los cuadernos. |
| `scripts/run_notebooks.py` | Ejecuta los cuadernos y reporta cuáles pasan. |
| `scripts/patch_notebooks.py` | Adapta cuadernos de Colab a local (ya aplicado; idempotente). |
