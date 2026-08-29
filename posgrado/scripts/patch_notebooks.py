"""Adapta los cuadernos de Colab para que también corran en local.

Cambios aplicados (todos preservan el comportamiento en Google Colab):

1. Celda de arranque que prepara el directorio de trabajo en la caché local.
2. Rutas absolutas ``/content/...`` convertidas en rutas relativas. En Colab el
   directorio de trabajo ya es ``/content``, así que el resultado es idéntico.
3. ``!git clone`` del repositorio del curso -> ``curso_setup.clonar_curso()``,
   que clona una sola vez y reutiliza la copia en caché.
4. Descargas de Google Drive con ``wget`` y cookies -> ``descargar_drive()``,
   que usa gdown, funciona en cualquier sistema operativo y cachea el archivo.
5. Instalaciones con ``!pip`` y utilidades exclusivas de Colab (cloudflared,
   streamlit en segundo plano) quedan condicionadas a ejecutarse solo en Colab.

Es idempotente: volver a ejecutarlo no duplica los cambios.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARCA = "# --- Arranque del entorno local"

ARRANQUE = '''# --- Arranque del entorno local (en Google Colab no cambia nada) ---
import pathlib
import sys
import types

try:
    _raiz = next(
        d
        for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
        if (d / "curso_setup.py").exists()
    )
    sys.path.insert(0, str(_raiz))
    import curso_setup
except StopIteration:  # Google Colab: se usa un sustituto mínimo
    import subprocess

    def _clonar(destino="curso_IA_CHEC"):
        if not pathlib.Path(destino).is_dir():
            subprocess.run(
                ["git", "clone", "https://github.com/UN-GCPDS/curso_IA_CHEC.git", destino],
                check=True,
            )
        return pathlib.Path(destino)

    def _descargar(file_id, destino):
        if not pathlib.Path(destino).exists():
            import gdown

            gdown.download(id=file_id, output=destino, quiet=False)
        return pathlib.Path(destino)

    curso_setup = types.SimpleNamespace(
        en_colab=lambda: True,
        init=lambda *a, **k: pathlib.Path.cwd(),
        clonar_curso=_clonar,
        descargar_drive=_descargar,
    )

curso_setup.init()'''

# Órdenes de shell que solo tienen sentido dentro de Colab.
SOLO_COLAB = re.compile(
    r"^(\s*)(!\s*(pip|apt|apt-get)\s|"
    r"!\s*wget\s+https://github\.com/cloudflare|"
    r"!\s*chmod\s|!\s*mv\s|!\s*streamlit\s|!\s*cloudflared\s|!\s*pkill\s|"
    r"%\s*pip\s)"
)

RE_CLONE = re.compile(r"^\s*!\s*git clone\s+\S*curso_IA_CHEC\.git[^\n]*$", re.M)
RE_DIR = re.compile(r"^\s*!\s*dir\s*$", re.M)
RE_WGET_DRIVE = re.compile(
    r"^[ \t]*!\s*wget\s+--load-cookies[\s\S]*?-O\s+(\S+)[\s\S]*?rm\s+-rf\s+/tmp/cookies\.txt[^\n]*$",
    re.M,
)
RE_CONTENT = re.compile(r"/content/")
# ChatOpenAI se movió a langchain_openai; el propio curso ya usa esa ruta en otro cuaderno.
RE_CHATOPENAI = re.compile(
    r"^from langchain_community\.chat_models import ChatOpenAI\s*$", re.M
)
# "%%capture" solo funciona en la PRIMERA línea de la celda. Cuando queda debajo
# de "# @title" IPython lo interpreta como magic de línea y falla, también en Colab.
RE_CAPTURE_HUERFANO = re.compile(r"(?<=\n)[ \t]*%%capture[ \t]*\n")
# device=0 / device='0' -> selección automática (CUDA en Windows/Linux, MPS en Mac)
RE_DEVICE = re.compile(r"""device\s*=\s*(?:0|['"]0['"]|['"]cuda['"]|['"]cuda:0['"])""")


# Celdas que solo tienen sentido dentro de Colab: túnel de Cloudflare y la app
# de Streamlit lanzada en segundo plano. Se protegen enteras porque también
# leen los logs que esas órdenes generan.
RE_TUNEL = re.compile(r"cloudflared|streamlit run", re.I)
# Definición de función en el margen izquierdo (nivel superior de la celda).
RE_DEF = re.compile(r"^(?:def|class)\s+\w+.*:\s*$")


def _guardar_solo_colab(fuente: str) -> str:
    """Condiciona a Colab las órdenes de shell que solo funcionan allí.

    Agrupa las líneas contiguas bajo un único ``if``. Es idempotente: si la
    celda ya está protegida no vuelve a envolverla (envolver dos veces anidaría
    los ``if`` en cada pasada).
    """
    if "curso_setup.en_colab()" in fuente:
        return fuente

    lineas = fuente.split("\n")
    if not any(SOLO_COLAB.match(l) for l in lineas):
        return fuente

    # Celda de túnel completa: se envuelve entera, salvo que empiece con un
    # cell magic (``%%writefile``), que no admite indentación.
    if RE_TUNEL.search(fuente) and not fuente.lstrip().startswith("%%"):
        # Si la celda define funciones no se puede envolver entera: la
        # definición quedaría dentro del ``if`` y, fuera de Colab, las celdas
        # que la invocan fallarían con NameError. Se protege el cuerpo.
        if any(RE_DEF.match(l) for l in lineas):
            salida: list[str] = []
            for l in lineas:
                salida.append(l)
                if RE_DEF.match(l):
                    salida.append("    if not curso_setup.en_colab():")
                    salida.append(
                        '        print("[curso] Esta celda solo funciona en Google Colab.")'
                    )
                    salida.append("        return")
            return "\n".join(salida)

        cuerpo = "\n".join(("    " + l) if l.strip() else l for l in lineas)
        return "if curso_setup.en_colab():\n" + cuerpo

    salida: list[str] = []
    i = 0
    while i < len(lineas):
        m = SOLO_COLAB.match(lineas[i])
        if not m:
            salida.append(lineas[i])
            i += 1
            continue
        sangria = m.group(1)
        salida.append(f"{sangria}if curso_setup.en_colab():")
        while i < len(lineas) and SOLO_COLAB.match(lineas[i]):
            salida.append(f"{sangria}    {lineas[i].strip()}")
            i += 1
    return "\n".join(salida)


def parchear(ruta: Path) -> list[str]:
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    celdas = nb.get("cells", [])
    cambios: list[str] = []

    celda_arranque = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["arranque-local"]},
        "outputs": [],
        # keepends es obligatorio: nbformat concatena la lista tal cual, y sin
        # los saltos de línea la celda entera colapsaría en una sola línea.
        "source": ARRANQUE.splitlines(keepends=True),
    }
    ya_tiene = celdas and MARCA in "".join(celdas[0].get("source", []))
    if ya_tiene:
        celdas[0] = celda_arranque
        cambios.append("celda de arranque (reescrita)")
    else:
        celdas.insert(0, celda_arranque)
        cambios.append("celda de arranque")

    for celda in celdas:
        if celda.get("cell_type") != "code":
            continue
        if "arranque-local" in celda.get("metadata", {}).get("tags", []):
            continue
        original = "".join(celda.get("source", []))
        s = original

        s = RE_CLONE.sub('curso_setup.clonar_curso("curso_IA_CHEC")', s)
        s = RE_WGET_DRIVE.sub(
            lambda m: f'curso_setup.descargar_drive(FILEID, "{m.group(1)}")', s
        )
        s = RE_CHATOPENAI.sub("from langchain_openai import ChatOpenAI", s)
        s = RE_CAPTURE_HUERFANO.sub("", s)
        s = RE_DEVICE.sub("device=curso_setup.dispositivo()", s)
        s = RE_DIR.sub("", s)
        s = RE_CONTENT.sub("", s)
        s = _guardar_solo_colab(s)

        if s != original:
            celda["source"] = s.splitlines(keepends=True)
            celda["outputs"] = []
            celda["execution_count"] = None

    if not cambios:
        cambios.append("rutas y órdenes de shell")

    # normalize() añade el campo "id" que nbformat >= 5.1.4 exige en cada celda
    try:
        import nbformat

        _, nb = nbformat.validator.normalize(nb, version=4, version_minor=5)
    except Exception:
        pass

    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cambios


def main() -> int:
    base = Path(__file__).resolve().parent.parent
    for ruta in sorted(base.glob("**/*.ipynb")):
        excluidos = {".cache", ".venv", "venv", ".git", ".ipynb_checkpoints"}
        if excluidos & set(ruta.parts):
            continue
        cambios = parchear(ruta)
        print(f"  {ruta.relative_to(base)}  -> {', '.join(cambios)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
