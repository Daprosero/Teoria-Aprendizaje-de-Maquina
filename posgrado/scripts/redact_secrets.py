"""Elimina credenciales incrustadas en los cuadernos.

Las claves venían escritas dentro del código del curso: una de OpenAI en un
comentario sobre el ``getpass`` y una de Roboflow como argumento literal. Este
script las reemplaza por un marcador y deja el cuaderno leyendo la credencial
desde una variable de entorno.

Es idempotente y sirve además como comprobación en CI:

    python scripts/redact_secrets.py            # aplica
    python scripts/redact_secrets.py --revisar  # solo informa; sale 1 si encuentra algo
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXCLUIDOS = {".cache", ".venv", "venv", ".git", ".ipynb_checkpoints"}

PATRONES = [
    # Clave de OpenAI dejada en un comentario encima del getpass.
    (re.compile(r"#[^\S\n]*sk-(?:proj-)?[A-Za-z0-9_\-]{20,}[^\S\n]*\n"), ""),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "<CLAVE-ELIMINADA>"),
    # api_key="..." de Roboflow -> se lee del entorno.
    (
        re.compile(r"""api_key\s*=\s*["'][A-Za-z0-9_\-]{15,}["']"""),
        'api_key=os.environ["ROBOFLOW_API_KEY"]',
    ),
]


def procesar(escribir: bool) -> int:
    encontrados = 0
    for ruta in sorted(BASE.glob("**/*.ipynb")):
        if EXCLUIDOS & set(ruta.parts):
            continue
        nb = json.loads(ruta.read_text(encoding="utf-8"))
        tocado = False
        for i, celda in enumerate(nb.get("cells", [])):
            if celda.get("cell_type") != "code":
                continue
            s = original = "".join(celda.get("source", []))
            for patron, reemplazo in PATRONES:
                s = patron.sub(reemplazo, s)
            if s != original:
                encontrados += 1
                print(f"  {ruta.relative_to(BASE)} [celda {i}]")
                if escribir:
                    celda["source"] = s.splitlines(keepends=True)
                    celda["outputs"] = []
                    tocado = True
        if tocado and escribir:
            ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return encontrados


def main() -> int:
    escribir = "--revisar" not in sys.argv
    print("Limpiando credenciales..." if escribir else "Revisando credenciales...")
    n = procesar(escribir)
    print(f"celdas afectadas: {n}")
    return 1 if (n and not escribir) else 0


if __name__ == "__main__":
    sys.exit(main())
