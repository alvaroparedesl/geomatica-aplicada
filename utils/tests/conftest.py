"""
Utilidades compartidas por las pruebas de utils/.

Los ítems STAC de `fixtures/items/` son ítems reales (recortados) de Planetary
Computer ("pc_*"), Earth Search sentinel-2-l2a ("es_*") y Earth Search
sentinel-2-c1-l2a ("c1_*") sobre Talcahuano (tile 18HXE). Se grabaron con
internet una sola vez, así las pruebas que los usan funcionan sin conexión.
"""

import json
from pathlib import Path

import pystac
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def leer_items(nombre):
    """Lista de pystac.Item desde fixtures/items/<nombre>.json (ej. 'pc_2021-12-14')."""
    datos = json.loads((FIXTURES / "items" / f"{nombre}.json").read_text())
    return [pystac.Item.from_dict(d) for d in datos]


def item_por_id(nombre, sufijo):
    """El ítem de fixtures/items/<nombre>.json cuyo id termina en `sufijo`."""
    (item,) = [i for i in leer_items(nombre) if i.id.endswith(sufijo)]
    return item


def todos_los_items():
    """Todos los ítems grabados, con el nombre del archivo de origen."""
    return [(ruta.stem, item) for ruta in sorted((FIXTURES / "items").glob("*.json")) for item in leer_items(ruta.stem)]


@pytest.fixture
def con_earthsearch_l2a(monkeypatch):
    """
    Reactiva el catálogo "earthsearch-l2a", desactivado para los estudiantes.

    Sus ítems y su regla de offset siguen probándose: la regla vive en
    `offset_sentinel2` y estas pruebas conservan lo que se midió en los píxeles.
    """
    from utils import stac

    monkeypatch.setitem(stac.CATALOGOS["earthsearch-l2a"], "desactivado", None)


@pytest.fixture
def items():
    return leer_items
