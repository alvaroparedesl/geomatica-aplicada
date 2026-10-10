"""
Qué demuestra este archivo (sin internet): la REGLA del offset de reflectancia
de Sentinel-2 y la limpieza de duplicados, con 33 ítems STAC reales grabados
en fixtures/items/ (metadatos de Planetary Computer, Earth Search l2a y c1 sobre
el tile 18HXE de Concepción, entre 2018 y 2026).

Regla (ver la sección "OFFSET DE REFLECTANCIA" de utils/stac.py):
* planetary (sentinel-2-l2a): -0.1 si baseline >= 04.00, si no 0.
* earthsearch (sentinel-2-c1-l2a): siempre -0.1.
* earthsearch-l2a (desactivado; la regla y sus pruebas se conservan): 0, salvo
  baseline >= 04.00, tomas del 25-01-2022 al 28-02-2022 y
  earthsearch:boa_offset_applied distinto de True (-0.1). La propiedad dice False
  en muchas imágenes posteriores que ya vienen sin el +1000 (comprobado en los
  píxeles, ver test_pixeles_reales.py), por eso sus metadatos no son confiables.

Cada fila de TABLA fija el offset de un ítem real: si cambias una regla de
`offset_sentinel2`, la fila correspondiente falla. También se prueban el
reconocimiento del catálogo de cada ítem, los límites de la ventana de Earth
Search l2a, el caso sin baseline y `quitar_duplicados`.
"""

import warnings

import pytest

from utils import stac

from .conftest import item_por_id, leer_items, todos_los_items

# (archivo, final del id, baseline, boa_offset_applied, offset esperado)
TABLA = [
    # Planetary Computer
    ("pc_2020-12-04", "20201205T163352", "02.12", None, 0.0),
    ("pc_2021-12-14", "20211215T041443", "03.00", None, 0.0),
    ("pc_2021-12-14", "20230129T131414", "04.00", None, -0.1),  # dic-2021 reprocesada
    ("pc_2022-01-23", "20220124T062622", "03.00", None, 0.0),  # enero 2022, aún baseline antiguo
    ("pc_2022-01-31", "20220216T205011", "04.00", None, -0.1),
    ("pc_2022-03-07", "20220308T061243", "04.00", None, -0.1),
    ("pc_2022-02-05", "20220219T034715", "04.00", None, -0.1),
    ("pc_2022-02-05", "20240520T155753", "05.10", None, -0.1),
    ("pc_2025-02-16", "20250216T180209", "05.11", None, -0.1),
    ("pc_2026-06-06", "20260607T021120", "05.12", None, -0.1),
    # Earth Search sentinel-2-l2a
    ("es_2018-04-17", "_0_L2A", "00.01", False, 0.0),
    ("es_2018-04-17", "_1_L2A", "00.01", None, 0.0),  # sin la propiedad, baseline antiguo
    ("es_2018-10-19", "_1_L2A", "00.01", False, 0.0),
    ("es_2018-10-19", "_2_L2A", "05.00", True, 0.0),
    ("es_2020-12-04", "_0_L2A", "02.14", False, 0.0),
    ("es_2020-12-04", "_1_L2A", "05.00", True, 0.0),
    ("es_2021-12-14", "_0_L2A", "03.01", False, 0.0),
    ("es_2021-12-14", "_1_L2A", "05.00", True, 0.0),
    ("es_2022-01-23", "_0_L2A", "03.01", False, 0.0),
    ("es_2022-01-31", "_0_L2A", "04.00", False, -0.1),  # ventana en que Earth Search no restó
    ("es_2022-02-25", "_0_L2A", "04.00", False, -0.1),  # último día con +1000 en Talcahuano
    ("es_2022-02-27", "_0_L2A", "04.00", True, 0.0),
    ("es_2022-03-07", "_0_L2A", "04.00", False, 0.0),  # dice False pero ya viene restado
    ("es_2022-03-19", "_0_L2A", "04.00", True, 0.0),
    ("es_2022-03-19", "_1_L2A", "05.10", False, 0.0),  # ídem
    ("es_2022-03-22", "_1_L2A", "05.10", False, 0.0),
    ("es_2024-11-11", "S2A_18HXE_20241111_0_L2A", "05.11", True, 0.0),
    ("es_2024-11-11", "S2C_18HXE_20241111_1_L2A", "05.11", False, 0.0),
    ("es_2025-02-16", "_0_L2A", "05.11", False, 0.0),  # ídem (ver test_pixeles_reales.py)
    ("es_2025-03-18", "_0_L2A", "05.11", True, 0.0),
    # Earth Search sentinel-2-c1-l2a
    ("c1_2020-12-04", "_L2A", "05.00", None, -0.1),
    ("c1_2021-12-14", "_L2A", "05.00", None, -0.1),
    ("c1_2025-02-16", "_L2A", "05.11", None, -0.1),
    ("c1_2025-03-18", "_L2A", "05.11", None, -0.1),
]


@pytest.mark.parametrize("archivo,sufijo,baseline,aplicado,esperado", TABLA)
def test_tabla_de_offsets(archivo, sufijo, baseline, aplicado, esperado):
    item = item_por_id(archivo, sufijo)
    assert stac.baseline_procesamiento(item) == baseline
    assert item.properties.get("earthsearch:boa_offset_applied") == aplicado
    assert stac.offset_sentinel2(item) == esperado
    assert stac.escala_y_offset(item, "B04") == (0.0001, esperado)


@pytest.mark.parametrize("nombre,item", todos_los_items(), ids=lambda v: getattr(v, "id", v))
def test_todos_los_items_tienen_offset_valido(nombre, item):
    """Ningún ítem grabado cae fuera de la regla (sin advertencias)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        offset = stac.offset_sentinel2(item)
    assert offset in (0.0, -0.1)
    if nombre.startswith("pc"):
        assert offset == (-0.1 if item.properties["s2:processing_baseline"] >= "04.00" else 0.0)
    if nombre.startswith("c1"):
        assert offset == -0.1


@pytest.mark.parametrize(
    "archivo,catalogo",
    [("pc_2022-01-31", "planetary"), ("es_2022-01-31", "earthsearch-l2a"), ("c1_2025-02-16", "earthsearch")],
)
def test_catalogo_de_item(archivo, catalogo):
    for item in leer_items(archivo):
        assert stac.catalogo_de_item(item) == catalogo


def test_catalogo_de_item_sin_enlaces():
    """Sin enlaces 'self' se reconoce por las URL de los archivos."""
    pc = leer_items("pc_2022-01-31")[0]
    es = leer_items("es_2022-01-31")[0]
    for item in (pc, es):
        item.links = []
    assert stac.catalogo_de_item(pc) == "planetary"
    assert stac.catalogo_de_item(es) == "earthsearch-l2a"


def test_metadatos_earthsearch_no_son_confiables():
    """
    Earth Search sentinel-2-l2a declara offset -0.1 en raster:bands (y a veces
    boa_offset_applied=False) aun cuando ya restó el desfase de los píxeles:
    por eso la regla no usa esos valores tal cual.
    """
    falso = item_por_id("es_2025-02-16", "_0_L2A")
    assert falso.properties["earthsearch:boa_offset_applied"] is False
    assert stac.offset_sentinel2(falso) == 0.0
    item = item_por_id("es_2025-03-18", "_0_L2A")
    banda = item.assets["red"].extra_fields["raster:bands"][0]
    assert banda["offset"] == -0.1 and item.properties["earthsearch:boa_offset_applied"] is True
    assert stac.offset_sentinel2(item) == 0.0
    # Planetary Computer no trae scale/offset en los metadatos
    pc = leer_items("pc_2025-02-16")[0]
    assert "raster:bands" not in pc.assets["B04"].extra_fields
    # c1: los metadatos coinciden con la regla
    c1 = leer_items("c1_2025-02-16")[0]
    assert c1.assets["red"].extra_fields["raster:bands"][0]["offset"] == stac.offset_sentinel2(c1) == -0.1


def test_sin_baseline_usa_la_fecha_y_avisa():
    antiguo = leer_items("pc_2020-12-04")[0]
    nuevo = leer_items("pc_2025-02-16")[0]
    for item in (antiguo, nuevo):
        del item.properties["s2:processing_baseline"]
    with pytest.warns(UserWarning, match="processing_baseline"):
        assert stac.offset_sentinel2(antiguo) == 0.0
    with pytest.warns(UserWarning, match="processing_baseline"):
        assert stac.offset_sentinel2(nuevo) == -0.1


def test_earthsearch_sin_propiedad_aplicada():
    """Sin la propiedad: -0.1 solo dentro de la ventana de febrero de 2022."""
    dentro = item_por_id("es_2022-01-31", "_0_L2A")
    fuera = item_por_id("es_2025-03-18", "_0_L2A")
    for item in (dentro, fuera):
        del item.properties["earthsearch:boa_offset_applied"]
    assert stac.offset_sentinel2(dentro) == -0.1
    assert stac.offset_sentinel2(fuera) == 0.0


@pytest.mark.parametrize("fecha,esperado", [("2022-01-24T15:00:00Z", 0.0), ("2022-01-25T15:00:00Z", -0.1),
                                            ("2022-02-28T15:00:00Z", -0.1), ("2022-03-01T15:00:00Z", 0.0)])
def test_limites_de_la_ventana_earthsearch(fecha, esperado):
    import datetime as dt

    item = item_por_id("es_2022-01-31", "_0_L2A")
    item.datetime = dt.datetime.fromisoformat(fecha.replace("Z", "+00:00"))
    assert stac.offset_sentinel2(item) == esperado
    item.properties["earthsearch:boa_offset_applied"] = True
    assert stac.offset_sentinel2(item) == 0.0


# ---------------------------------------------------------------- duplicados
@pytest.mark.parametrize(
    "archivo,quedan",
    [
        ("pc_2021-12-14", ["20230129T131414"]),  # 03.00 y 04.00: queda 04.00
        ("pc_2022-02-05", ["20240520T155753"]),  # 04.00 y 05.10: queda 05.10
        ("pc_2022-04-28", ["20220714T000357"]),  # mismo baseline: la generada después
        ("pc_2026-06-06", ["20260607T021120", "20260606T181755"]),  # dos satélites: ambas
        ("es_2018-04-17", ["_1_L2A"]),  # mismo baseline, la generada después
        ("es_2018-10-19", ["_1_L2A", "_2_L2A"]),  # dos datastrips: uno reprocesado (05.00) y otro no
        ("es_2020-12-04", ["_1_L2A"]),
        ("es_2021-12-14", ["_1_L2A"]),
        ("es_2022-03-19", ["_1_L2A"]),
        ("es_2022-03-22", ["_1_L2A"]),
        ("es_2024-11-11", ["S2A_18HXE_20241111_0_L2A", "S2C_18HXE_20241111_1_L2A"]),
        ("es_2026-04-27", ["S2A_18HXE_20260427_3_L2A", "S2B_18HXE_20260427_0_L2A"]),  # copias idénticas
        ("c1_2024-11-11", ["S2A_T18HXE_20241111T145640_L2A", "S2C_T18HXE_20241111T145855_L2A"]),
        ("c1_2025-02-16", ["_L2A"]),
    ],
)
def test_quitar_duplicados(archivo, quedan):
    items = leer_items(archivo)
    resultado = stac.quitar_duplicados(items)
    assert len(resultado) == len(quedan)
    for sufijo in quedan:
        assert sum(item.id.endswith(sufijo) for item in resultado) == 1
    # idempotente y conserva el orden original
    assert stac.quitar_duplicados(resultado) == resultado
    assert resultado == [item for item in items if item in resultado]


def test_quitar_duplicados_no_mezcla_catalogos_ni_toca_otros_items():
    pc = leer_items("pc_2021-12-14")
    es = leer_items("es_2021-12-14")
    otro = leer_items("pc_2020-12-04")[0].clone()
    otro.collection_id = "landsat-c2-l2"
    resultado = stac.quitar_duplicados(pc + es + [otro, otro])
    assert [i.id for i in resultado] == [
        "S2A_MSIL2A_20211214T143731_R096_T18HXE_20230129T131414",
        "S2A_18HXE_20211214_1_L2A",
        otro.id,
        otro.id,
    ]


def test_quitar_duplicados_informa(capsys):
    stac.quitar_duplicados(leer_items("es_2018-10-19"), verbose=True)
    assert "Se quitaron 1 versiones duplicadas" in capsys.readouterr().out


def test_tras_quitar_duplicados_no_hay_mezcla_de_baselines_por_toma():
    """Después de limpiar, cada (catálogo, satélite, tile, datastrip) aparece una vez."""
    items = [item for _, item in todos_los_items()]
    limpios = stac.quitar_duplicados(items)
    claves = [
        (stac.catalogo_de_item(i), i.properties["platform"].lower(), stac._tile(i), stac._inicio_datastrip(i))
        for i in limpios
    ]
    assert len(claves) == len(set(claves))
    # y ninguno de los descartados tenía un baseline mayor que el conservado
    for item in items:
        clave = (stac.catalogo_de_item(item), item.properties["platform"].lower(), stac._tile(item), stac._inicio_datastrip(item))
        elegido = limpios[claves.index(clave)]
        assert stac._version(stac.baseline_procesamiento(item)) <= stac._version(stac.baseline_procesamiento(elegido))


# ---------------------------------------------------------------- resumen y búsqueda
def test_resumir_imagenes_sentinel2_muestra_baseline_y_offset():
    items = stac.quitar_duplicados(leer_items("es_2022-03-19") + leer_items("es_2025-03-18"))
    tabla = stac.resumir_imagenes(items)
    assert list(tabla.columns) == ["fecha", "nubes_%", "id", "baseline", "offset"]
    assert tabla["offset"].tolist() == [0.0, 0.0]
    assert tabla["baseline"].tolist() == ["05.10", "05.11"]


def test_nombres_de_catalogo():
    assert stac._nombre_catalogo(None) == stac.CATALOGO_SENTINEL2
    assert stac._nombre_catalogo("earthsearch") == "earthsearch"
    with pytest.raises(ValueError, match="no reconocido"):
        stac._nombre_catalogo("google")
    with pytest.raises(ValueError, match="no reconocido"):
        stac._nombre_catalogo("earthsearch-c1")  # el nombre antiguo de "earthsearch"
    with pytest.raises(ValueError, match="desactivado"):
        stac._nombre_catalogo("earthsearch-l2a")
    assert set(stac.CATALOGOS) == {"planetary", "earthsearch", "earthsearch-l2a"}
    assert stac.CATALOGOS["earthsearch"]["colecciones"]["sentinel-2"] == "sentinel-2-c1-l2a"
    assert stac.CATALOGO_SENTINEL2 == "planetary"  # por ahora el curso usa Planetary Computer


def test_bandas_earthsearch_cubren_las_bandas_del_curso():
    from utils import raster

    for banda in list(raster.SENSORES["sentinel-2"]["bandas"].values()) + ["SCL"]:
        assert banda in stac.BANDAS_EARTHSEARCH
    assert len(set(stac.BANDAS_EARTHSEARCH.values())) == len(stac.BANDAS_EARTHSEARCH)
    # los nombres de Earth Search existen en los ítems grabados
    item = leer_items("es_2025-02-16")[0]
    for banda in ("B02", "B04", "B08", "SCL"):
        assert stac.BANDAS_EARTHSEARCH[banda] in item.assets
    c1 = leer_items("c1_2025-02-16")[0]
    for banda in ("B02", "B04", "B08", "SCL"):
        assert stac.BANDAS_EARTHSEARCH[banda] in c1.assets
