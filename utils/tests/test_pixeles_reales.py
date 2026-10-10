"""
Qué demuestra este archivo (sin internet): que la regla de `offset_sentinel2`
acierta sobre PÍXELES reales, no solo sobre metadatos.

Usa DN reales de la misma toma en Planetary Computer, Earth Search
sentinel-2-l2a (desactivado, pero su regla se conserva) y Earth Search
sentinel-2-c1-l2a: ventanas pequeñas de 15 escenas grabadas en
fixtures/pixeles/*.npz, con las bandas B02, B04, B08 y SCL.

Los DN de cada catálogo son distintos (con o sin el +1000), pero después de
aplicar `offset_sentinel2` la reflectancia debe coincidir: el sesgo (mediana de
las diferencias píxel a píxel) debe ser < 0.005, cuando un error de offset da
0.1. Píxel a píxel hay diferencias algo mayores en tierra (NIR ~0.01) porque
las vistas reducidas (overviews) de cada catálogo se calcularon distinto. Las áreas son mar
abierto al oeste de Hualpén ("oceano", agua oscura: donde un error de offset
se nota más) y tierra en Hualpén ("tierra").
"""

from pathlib import Path

import numpy as np
import pytest

from utils import stac

from .conftest import FIXTURES, item_por_id

# toma -> [(archivo de ítems, sufijo del id)]; el primero es la referencia
TOMAS = {
    "2021-12-14": [("pc_2021-12-14", "20230129T131414"), ("pc_2021-12-14", "20211215T041443"),
                   ("es_2021-12-14", "_1_L2A"), ("es_2021-12-14", "_0_L2A"), ("c1_2021-12-14", "_L2A")],
    "2022-01-31": [("pc_2022-01-31", "T205011"), ("es_2022-01-31", "_0_L2A")],
    "2022-03-07": [("pc_2022-03-07", "T061243"), ("es_2022-03-07", "_0_L2A")],  # ES: dice False, viene restado
    "2025-02-16": [("pc_2025-02-16", "T180209"), ("es_2025-02-16", "_0_L2A"), ("c1_2025-02-16", "_L2A")],
    "2025-03-18": [("pc_2025-03-18", "T191713"), ("es_2025-03-18", "_0_L2A"), ("c1_2025-03-18", "_L2A")],
}
CASOS = [(fecha, archivo, sufijo) for fecha, lista in TOMAS.items() for archivo, sufijo in lista[1:]]


def _datos(archivo, sufijo):
    item = item_por_id(archivo, sufijo)
    ruta = FIXTURES / "pixeles" / f"{item.id}.npz"
    return item, dict(np.load(ruta))


def _reflectancia(dn, offset):
    dn = dn.astype("float64")
    return np.where(dn > 0, dn * 0.0001 + offset, np.nan)


def _mascara(area, ref, otro):
    clases = [6] if area == "oceano" else [4, 5]
    return np.isin(ref[f"{area}_SCL"], clases) & np.isin(otro[f"{area}_SCL"], clases)


@pytest.mark.parametrize("fecha,archivo,sufijo", CASOS, ids=[f"{c[0]}-{c[1][:2]}{c[2]}" for c in CASOS])
@pytest.mark.parametrize("area", ["oceano", "tierra"])
def test_misma_reflectancia_entre_catalogos(fecha, archivo, sufijo, area):
    item_ref, ref = _datos(*TOMAS[fecha][0])
    item, otro = _datos(archivo, sufijo)
    mascara = _mascara(area, ref, otro)
    assert mascara.sum() > 50, f"pocos píxeles de {area}"
    for banda in ("B02", "B04", "B08"):
        a = _reflectancia(ref[f"{area}_{banda}"], stac.offset_sentinel2(item_ref))[mascara]
        b = _reflectancia(otro[f"{area}_{banda}"], stac.offset_sentinel2(item))[mascara]
        ok = np.isfinite(a) & np.isfinite(b)
        sesgo = np.median(b[ok] - a[ok])
        assert abs(sesgo) < 0.005, f"{item.id} {banda} {area}: sesgo = {sesgo:+.4f}"
        ruido = np.median(np.abs(b[ok] - a[ok]))
        assert ruido < (0.005 if area == "oceano" else 0.025), f"{item.id} {banda} {area}: |dif| = {ruido:.4f}"


@pytest.mark.parametrize("fecha,archivo,sufijo", CASOS + [(f, *TOMAS[f][0]) for f in TOMAS],
                         ids=lambda v: str(v)[:20])
def test_agua_oscura_pero_no_negativa(fecha, archivo, sufijo):
    item, datos = _datos(archivo, sufijo)
    agua = datos["oceano_SCL"] == 6
    for banda in ("B02", "B04", "B08"):
        refl = _reflectancia(datos[f"oceano_{banda}"], stac.offset_sentinel2(item))[agua]
        assert 0 <= np.nanmedian(refl) < 0.05, f"{item.id} {banda}"
        assert np.mean(refl < 0) < 0.05  # casi ningún píxel negativo


@pytest.mark.parametrize("archivo,sufijo,offset_malo", [
    ("pc_2021-12-14", "20211215T041443", -0.1),  # el -0.1 fijo de antes, en una imagen 03.00
    ("pc_2021-12-14", "20230129T131414", 0.0),   # sin offset en la reprocesada 04.00
    ("es_2025-03-18", "_0_L2A", -0.1),           # el offset de los metadatos de Earth Search
    ("es_2025-02-16", "_0_L2A", -0.1),           # creerle a boa_offset_applied=False
    ("es_2022-03-07", "_0_L2A", -0.1),           # ídem
    ("es_2022-01-31", "_0_L2A", 0.0),            # olvidar la ventana de febrero de 2022
])
def test_un_offset_equivocado_se_detecta(archivo, sufijo, offset_malo):
    """Control: las pruebas anteriores fallarían con el offset equivocado."""
    item, datos = _datos(archivo, sufijo)
    agua = datos["oceano_SCL"] == 6
    bien = _reflectancia(datos["oceano_B04"], stac.offset_sentinel2(item))[agua]
    mal = _reflectancia(datos["oceano_B04"], offset_malo)[agua]
    assert abs(np.nanmedian(mal) - np.nanmedian(bien)) == pytest.approx(0.1, abs=1e-6)
    assert not (0 <= np.nanmedian(mal) < 0.05)


def test_dn_crudos_reflejan_cada_convencion():
    """Agua oscura: DN ~1000 más altos donde el catálogo no restó el desfase."""
    medianas = {}
    for archivo, sufijo in TOMAS["2025-02-16"] + TOMAS["2025-03-18"]:
        item, datos = _datos(archivo, sufijo)
        medianas[item.id] = (stac.offset_sentinel2(item), np.median(datos["oceano_B04"][datos["oceano_SCL"] == 6]))
    for offset, mediana in medianas.values():
        assert (mediana > 1000) if offset == -0.1 else (mediana < 1000)


def test_fixtures_presentes():
    for lista in TOMAS.values():
        for archivo, sufijo in lista:
            item = item_por_id(archivo, sufijo)
            assert Path(FIXTURES / "pixeles" / f"{item.id}.npz").exists(), item.id
