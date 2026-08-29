# Teoría de Aprendizaje de Máquina

Material del curso: notebooks, prácticas y presentaciones.

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

Pendiente de organizar.

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
