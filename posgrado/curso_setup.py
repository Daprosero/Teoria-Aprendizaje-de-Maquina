"""Arranque local para los cuadernos del curso.

En Google Colab este módulo no altera nada: los cuadernos siguen ejecutándose
exactamente igual que antes.

Fuera de Colab prepara un directorio de trabajo equivalente a ``/content``
dentro de la caché local (``posgrado/.cache/content``), que está declarada en
``.gitignore``. Todo lo que los cuadernos clonan o descargan (repositorios de
GitHub, archivos de Google Drive, datasets de Kaggle, pesos de modelos) queda
allí y nunca se sube al repositorio.

Uso dentro de un cuaderno::

    import curso_setup; curso_setup.init()

La ubicación de la caché se puede cambiar con la variable de entorno
``CURSO_CACHE``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_CURSO = "https://github.com/UN-GCPDS/curso_IA_CHEC.git"

_RAIZ_MODULO = Path(__file__).resolve().parent


def en_colab() -> bool:
    """Indica si el cuaderno se está ejecutando en Google Colab."""
    return "google.colab" in sys.modules


def raiz_cache() -> Path:
    """Directorio raíz de la caché local (configurable con ``CURSO_CACHE``)."""
    return Path(os.environ.get("CURSO_CACHE", _RAIZ_MODULO / ".cache")).expanduser()


def init(silencioso: bool = False) -> Path:
    """Prepara el directorio de trabajo y devuelve su ruta.

    En Colab devuelve ``/content`` sin tocar nada. Fuera de Colab crea
    ``<cache>/content``, redirige las cachés de terceros y hace ``chdir``.
    """
    if en_colab():
        return Path("/content")

    cache = raiz_cache()
    trabajo = cache / "content"
    trabajo.mkdir(parents=True, exist_ok=True)

    # Redirigir las cachés de librerías de terceros dentro de .cache/
    for variable, subruta in (
        ("PIP_CACHE_DIR", "pip"),
        ("KAGGLEHUB_CACHE", "kagglehub"),
        ("HF_HOME", "hf"),
        ("TORCH_HOME", "torch"),
        ("YOLO_CONFIG_DIR", "ultralytics"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        destino = cache / subruta
        destino.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destino)

    # Evitar que Matplotlib abra ventanas al ejecutar sin interfaz gráfica
    os.environ.setdefault("MPLBACKEND", "Agg")
    # En Apple Silicon, delegar a CPU las operaciones que MPS todavía no cubre
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if _RAIZ_MODULO.as_posix() not in sys.path:
        sys.path.insert(0, _RAIZ_MODULO.as_posix())

    _vincular_recursos(Path.cwd(), trabajo)
    os.chdir(trabajo)
    if not silencioso:
        print(f"[curso] Directorio de trabajo: {trabajo}")
    return trabajo


_EXTENSIONES_RECURSO = {
    ".pkl", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml",
    ".jpg", ".jpeg", ".png", ".zip", ".txt", ".pdf", ".npy",
}


def _vincular_recursos(origen: Path, trabajo: Path) -> None:
    """Enlaza los archivos de datos que viven junto al cuaderno.

    Así los cuadernos siguen refiriéndose a ``Eventos_transformador.pkl`` y
    similares con rutas relativas, sin copiar nada al directorio de trabajo.
    """
    if origen == trabajo:
        return
    for archivo in origen.iterdir():
        if not archivo.is_file() or archivo.suffix.lower() not in _EXTENSIONES_RECURSO:
            continue
        enlace = trabajo / archivo.name
        if enlace.exists() or enlace.is_symlink():
            continue
        try:
            enlace.symlink_to(archivo.resolve())
        except OSError:
            pass


def clonar_curso(destino: str = "curso_IA_CHEC") -> Path:
    """Clona el repositorio de datos del curso una sola vez y lo reutiliza."""
    ruta = Path(destino)
    if ruta.is_dir():
        print(f"[curso] '{destino}' ya está en caché, no se vuelve a clonar.")
        return ruta.resolve()
    print(f"[curso] Clonando {REPO_CURSO} en {ruta.resolve()} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_CURSO, str(ruta)],
        check=True,
    )
    return ruta.resolve()


def descargar_drive(file_id: str, destino: str) -> Path:
    """Descarga un archivo de Google Drive una sola vez y lo reutiliza.

    Reemplaza las invocaciones a ``wget`` con cookies, que dependen de
    utilidades de GNU y no funcionan fuera de Linux.
    """
    ruta = Path(destino)
    if ruta.exists() and ruta.stat().st_size > 0:
        print(f"[curso] '{destino}' ya está en caché.")
        return ruta.resolve()
    import gdown

    print(f"[curso] Descargando '{destino}' desde Google Drive ...")
    gdown.download(id=file_id, output=str(ruta), quiet=False)
    return ruta.resolve()


def dispositivo():
    """Devuelve el acelerador disponible en la máquina actual.

    - ``0``      -> GPU NVIDIA vía CUDA (Windows y Linux, y Colab con GPU).
    - ``"mps"``  -> GPU de Apple Silicon vía Metal (macOS).
    - ``"cpu"``  -> sin acelerador.

    El formato es el que esperan Ultralytics/YOLO y PyTorch, de modo que el
    mismo cuaderno corre en Windows, Linux y macOS sin editar nada.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def dispositivo_torch():
    """Igual que :func:`dispositivo`, pero como ``torch.device``."""
    import torch

    d = dispositivo()
    return torch.device("cuda:0" if d == 0 else d)
