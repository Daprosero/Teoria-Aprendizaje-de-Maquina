# Posgrado — Teoría de Aprendizaje de Máquina

## 📊 Diapositivas del curso

**https://daprosero.github.io/Teoria-Aprendizaje-de-Maquina/**

Seis módulos con la teoría y los ejemplos de estas clases:

| Módulo | Contenido |
|--------|-----------|
| 1. Datos: de crudos a utilizables | `.info()` y `.describe()` como diagnóstico, mecanismos de datos faltantes, las cinco estrategias de imputación, fuga de información y escalado |
| 2. Gráficas y estadística descriptiva | Histograma, boxplot, violín, dispersión, barras, ECDF y matriz de correlación: qué pregunta responde cada uno y cómo se interpreta |
| 3. Regresión | Dependencia estadística, los tres enfoques de estimación, diagnóstico por residuos, sesgo-varianza, Ridge/Lasso/ElasticNet, métricas y validación cruzada |
| 4. Clasificación | Fronteras de decisión, k-vecinos, árboles, bosques aleatorios, SVM, matriz de confusión y elección del umbral |
| 5. Conglomerados | K-means y su supuesto oculto, elección de k, DBSCAN, espectral y validación de grupos |
| 6. Reducción de dimensión | Maldición de la dimensión, PCA, varianza explicada, kernel PCA, t-SNE y UMAP |

El orden es supervisado antes que no supervisado: PCA y UMAP llegan cuando ya se
sabe qué es entrenar y medir error.

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

## Limitaciones conocidas fuera de Colab

| Cuaderno | Limitación |
|----------|------------|
| `Clase 2/Conglomerados y Reduccion de Dimenciones Práctica.ipynb`, `Práctica_Conglomerados-Reducción.ipynb` | Usan `cuml` (RAPIDS de NVIDIA): solo Linux con GPU CUDA. En macOS no hay equivalente instalable. |
| `Clase 4/2_Detección_de_Objetos.ipynb` | Requiere claves de API de Roboflow y Weights & Biases. El dispositivo ya se elige solo (CUDA/MPS/CPU). |
| `Clase 4/Agentes_Inteligentes.ipynb`, `Clase 4/Untitled0.ipynb` | Requieren `OPENAI_API_KEY`. |
| `Clase 1/Práctica - procesamiento de datos.ipynb`, `Práctica_Conglomerados-Reducción.ipynb` | Requieren credenciales de Kaggle (`~/.kaggle/kaggle.json`). |
| Celdas de `streamlit` + `cloudflared` | Son túneles propios de Colab. Quedan condicionadas a Colab; en local se ejecuta la app con `streamlit run <archivo>.py`. |

## Scripts

| Script | Para qué sirve |
|--------|----------------|
| `curso_setup.py` | Prepara el directorio de trabajo y las cachés. Lo llaman los cuadernos. |
| `scripts/run_notebooks.py` | Ejecuta los cuadernos y reporta cuáles pasan. |
| `scripts/patch_notebooks.py` | Adapta cuadernos de Colab a local (ya aplicado; idempotente). |
