"""Comprueba la lógica de los cuadernos que necesitan una API key.

No usa una clave real: inyecta una ficticia y ejecuta el cuaderno celda por
celda registrando hasta dónde llega. Sirve para distinguir dos cosas muy
distintas:

  * el cuaderno está bien y solo le falta la credencial  -> falla en la llamada
    a la API con un error de autenticación;
  * el cuaderno tiene un error de lógica                 -> falla antes, en un
    import, en la construcción del agente o en los datos.

Uso:  python scripts/check_logica_agentes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

BASE = Path(__file__).resolve().parent.parent

CUADERNOS = [
    "Clase 4/Agentes_Inteligentes.ipynb",
    "Clase 4/Untitled0.ipynb",
]

# Neutraliza getpass y fija una clave ficticia, para que el cuaderno avance
# hasta la primera llamada real a la API.
INYECCION = '''import getpass as _getpass, os as _os
_os.environ.setdefault("OPENAI_API_KEY", "sk-clave-ficticia-solo-para-validar")
_getpass.getpass = lambda *a, **k: _os.environ["OPENAI_API_KEY"]
'''

# Errores que significan "la lógica está bien, solo falta la credencial".
SENALES_DE_CREDENCIAL = (
    "AuthenticationError",
    "Incorrect API key",
    "invalid_api_key",
    "401",
    "RateLimitError",
    "insufficient_quota",
)


def revisar(ruta_rel: str) -> None:
    ruta = BASE / ruta_rel
    nb = nbformat.read(ruta, as_version=4)

    inyectada = nbformat.v4.new_code_cell(INYECCION)
    nb.cells.insert(1, inyectada)  # después del arranque del entorno

    cliente = NotebookClient(
        nb,
        timeout=180,
        kernel_name="curso-posgrado",
        resources={"metadata": {"path": str(ruta.parent)}},
        allow_errors=True,   # queremos ver TODAS las celdas, no parar en la primera
    )
    cliente.execute()

    total = sum(1 for c in nb.cells if c.cell_type == "code")
    fallos = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                traza = "\n".join(salida.get("traceback", []))
                fallos.append((i, salida.get("ename", ""), salida.get("evalue", "")[:150], traza))

    print(f"\n=== {ruta_rel} ===")
    print(f"celdas de código: {total} | celdas con error: {len(fallos)}")
    if not fallos:
        print("  Sin errores: la lógica corre entera.")
        return
    for i, ename, evalue, traza in fallos:
        credencial = any(s in ename or s in evalue or s in traza for s in SENALES_DE_CREDENCIAL)
        etiqueta = "SOLO FALTA LA CREDENCIAL" if credencial else "ERROR DE LÓGICA"
        print(f"  [celda {i}] {etiqueta}")
        print(f"      {ename}: {evalue}")


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    for c in CUADERNOS:
        revisar(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
