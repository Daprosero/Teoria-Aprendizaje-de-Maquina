"""Correcciones de contenido de los cuadernos, separadas del entorno.

Cada corrección se documenta con el síntoma y el porqué. Este archivo existe
aparte de ``patch_notebooks.py`` para que quede claro qué toca el entorno y qué
toca el material del curso. Es idempotente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# El cuaderno construye 'titanic_copy' y el callback del widget usa 'df_2', que
# no se define en ninguna parte: es un renombrado que quedó a medias. Sin esto,
# las celdas posteriores fallan con NameError incluso en Colab.
FIX_DF2 = (
    "Clase 1/Práctica - procesamiento de datos.ipynb",
    "titanic_copy = titanic.copy()",
    "titanic_copy = titanic.copy()\ndf_2 = titanic_copy  # alias: el resto del cuaderno se refiere a df_2",
)


def aplicar(ruta_rel: str, viejo: str, nuevo: str) -> bool:
    ruta = BASE / ruta_rel
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    cambiado = False
    for celda in nb.get("cells", []):
        if celda.get("cell_type") != "code":
            continue
        s = "".join(celda.get("source", []))
        if viejo in s and nuevo not in s:
            celda["source"] = s.replace(viejo, nuevo).splitlines(keepends=True)
            cambiado = True
    if cambiado:
        ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cambiado


def main() -> int:
    ruta, viejo, nuevo = FIX_DF2
    print(f"  {ruta}: {'aplicado' if aplicar(ruta, viejo, nuevo) else 'ya estaba'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
