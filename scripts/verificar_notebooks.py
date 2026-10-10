#!/usr/bin/env python
"""Verifica que cada cuaderno jupytext (.py) tenga su .ipynb al día.

Recorre `notebooks/` y `talleres/`, y para cada `.py` en formato jupytext
(`py:percent`) comprueba que exista el `.ipynb` hermano y que el tipo y el
texto de cada celda coincidan con los del `.py`. Las salidas y los metadatos
se ignoran: solo se comparan las fuentes de las celdas.

Uso:
    python scripts/verificar_notebooks.py

Termina con código 1 si algún cuaderno falta o está desactualizado, e indica
el comando para regenerarlo.
"""

import sys
from pathlib import Path

import jupytext

RAIZ = Path(__file__).resolve().parent.parent
CARPETAS = ("notebooks", "talleres")


def es_jupytext(ruta):
    """True si el .py tiene la cabecera jupytext (los .py sueltos se ignoran)."""
    with ruta.open(encoding="utf-8") as f:
        cabecera = f.read(2000)
    return "jupytext:" in cabecera or "# %%" in cabecera


def celdas(nb):
    """Lista de (tipo, fuente) de un cuaderno, sin salidas ni metadatos."""
    return [(c.cell_type, c.source.strip()) for c in nb.cells]


def comparar(py):
    """Devuelve None si el .ipynb está al día, o un texto con el problema."""
    ipynb = py.with_suffix(".ipynb")
    if not ipynb.exists():
        return "no existe el .ipynb"
    esperado = celdas(jupytext.read(py, fmt="py:percent"))
    actual = celdas(jupytext.read(ipynb))
    if len(esperado) != len(actual):
        return f"el .ipynb tiene {len(actual)} celdas y el .py {len(esperado)}"
    for i, (e, a) in enumerate(zip(esperado, actual), start=1):
        if e != a:
            return f"la celda {i} ({e[0]}) difiere entre el .py y el .ipynb"
    return None


def main():
    fallos = []
    for carpeta in CARPETAS:
        base = RAIZ / carpeta
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            if not es_jupytext(py):
                continue
            problema = comparar(py)
            if problema:
                fallos.append((py.relative_to(RAIZ), problema))

    if not fallos:
        return 0

    print("Hay cuadernos .ipynb desactualizados respecto de su fuente .py:\n")
    for ruta, problema in fallos:
        print(f"  {ruta}: {problema}")
        print(f"    arreglo: jupytext --to notebook {ruta}")
    print("\nEl .py es la fuente: edítalo y regenera el .ipynb con jupytext.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
