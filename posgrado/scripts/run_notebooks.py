"""Ejecuta todos los cuadernos del curso y reporta el resultado.

Cada cuaderno se ejecuta en su propio kernel, con el directorio del cuaderno
como directorio inicial. La copia ejecutada se guarda en la caché para no
inflar el repositorio con salidas nuevas.

    python scripts/run_notebooks.py [patrón ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

BASE = Path(__file__).resolve().parent.parent
EXCLUIDOS = {".cache", ".venv", "venv", ".git", ".ipynb_checkpoints", "scripts"}
TIEMPO_CELDA = int(os.environ.get("TIEMPO_CELDA", "600"))


def cuadernos(patrones: list[str]) -> list[Path]:
    todos = [p for p in sorted(BASE.glob("**/*.ipynb")) if not (EXCLUIDOS & set(p.parts))]
    if not patrones:
        return todos
    return [p for p in todos if any(x.lower() in str(p).lower() for x in patrones)]


def ejecutar(ruta: Path) -> dict:
    nb = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(
        nb,
        timeout=TIEMPO_CELDA,
        kernel_name=os.environ.get("KERNEL_CURSO", "curso-posgrado"),
        resources={"metadata": {"path": str(ruta.parent)}},
        allow_errors=False,
    )
    inicio = time.time()
    resultado = {"cuaderno": str(ruta.relative_to(BASE)), "estado": "OK"}
    try:
        cliente.execute()
    except CellExecutionError as e:
        resultado["estado"] = "ERROR"
        resultado["error"] = f"{e.ename}: {e.evalue}"
        for i, celda in enumerate(nb.cells):
            if any(s.get("output_type") == "error" for s in celda.get("outputs", [])):
                resultado["celda"] = i
                break
    except Exception as e:  # timeout del kernel, kernel muerto, etc.
        resultado["estado"] = "ERROR"
        resultado["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        resultado["segundos"] = round(time.time() - inicio, 1)
        salida = BASE / ".cache" / "ejecutados" / ruta.relative_to(BASE)
        salida.parent.mkdir(parents=True, exist_ok=True)
        try:
            nbformat.write(nb, salida)
        except Exception:
            pass
    return resultado


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    lista = cuadernos(sys.argv[1:])
    print(f"Ejecutando {len(lista)} cuadernos (límite {TIEMPO_CELDA}s por celda)\n")
    resultados = []
    for ruta in lista:
        etiqueta = str(ruta.relative_to(BASE))
        print(f"→ {etiqueta} ...", flush=True)
        r = ejecutar(ruta)
        resultados.append(r)
        marca = "✓" if r["estado"] == "OK" else "✗"
        detalle = "" if r["estado"] == "OK" else f"  celda {r.get('celda','?')}: {r.get('error','')[:160]}"
        print(f"  {marca} {r['estado']}  ({r['segundos']}s){detalle}\n", flush=True)

    informe = BASE / ".cache" / "informe_ejecucion.json"
    informe.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in resultados if r["estado"] == "OK"]
    print("=" * 70)
    print(f"RESULTADO: {len(ok)}/{len(resultados)} cuadernos sin errores")
    print("=" * 70)
    for r in resultados:
        marca = "✓" if r["estado"] == "OK" else "✗"
        print(f"{marca} {r['cuaderno']}")
        if r["estado"] != "OK":
            print(f"    celda {r.get('celda','?')} — {r.get('error','')[:220]}")
    print(f"\nInforme completo: {informe}")
    return 0 if len(ok) == len(resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
