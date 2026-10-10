"""
Qué demuestra este archivo (sin internet): la parte de BÚSQUEDA de utils.stac.

* `parametros_busqueda` solo arma el diccionario para `catalogo.search(**...)`
  (colección, bbox, fechas y filtro `query`) y no ejecuta ninguna búsqueda.
* La elección de catálogo: "planetary" por defecto, "earthsearch" es la
  Colección 1 (sentinel-2-c1-l2a), "earthsearch-l2a" está DESACTIVADO (ValueError)
  y Landsat siempre es de Planetary Computer.
* El aviso por el hueco de Earth Search c1 (casi sin imágenes entre el 29-12-2021 y el 07-12-2022).
* `preparar_items`: quita duplicados, rechaza catálogos mezclados o desactivados
  y ordena por fecha.

La búsqueda real contra los catálogos se prueba en test_red_catalogos.py (marca `red`).
"""

import warnings

import pystac
import pytest

from utils import stac

from .conftest import leer_items

BBOX = [-73.2, -36.8, -73.1, -36.7]


class ClienteFalso:
    """Imita a pystac_client.Client: se reconoce por su URL y guarda los argumentos de `search`."""

    def __init__(self, url):
        self.url = url
        self.llamadas = []

    def get_self_href(self):
        return self.url

    def search(self, **kwargs):
        self.llamadas.append(kwargs)
        return kwargs


# ---------------------------------------------------------------- parametros_busqueda
def test_parametros_por_defecto_en_planetary():
    parametros = stac.parametros_busqueda("sentinel-2", BBOX, ("2021-12-14", "2021-12-15"), nubes_max=30)
    assert parametros == {
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX,
        "datetime": "2021-12-14/2021-12-15",
        "query": {"eo:cloud_cover": {"lt": 30}},  # `query`, no `filter` (Earth Search ignora CQL2)
    }
    assert "filter" not in parametros


def test_fechas_como_texto_o_pareja():
    a = stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-02-28"))
    b = stac.parametros_busqueda("sentinel-2", BBOX, "2024-01-01/2024-02-28")
    assert a == b and a["datetime"] == "2024-01-01/2024-02-28"
    with pytest.raises(ValueError, match="fechas"):
        stac.parametros_busqueda("sentinel-2", BBOX, "2024-01-01")


def test_no_ejecuta_la_busqueda_y_sirve_para_search_nativo():
    """Los parámetros se pasan a `catalogo.search(**parametros)`: la llamada nativa la hace el estudiante."""
    cliente = ClienteFalso(stac.URL_PLANETARY)
    parametros = stac.parametros_busqueda("sentinel-2", BBOX, ("2021-12-14", "2021-12-14"), catalogo=cliente)
    assert cliente.llamadas == []  # armar los parámetros no busca nada
    assert cliente.search(**parametros)["collections"] == ["sentinel-2-l2a"]
    assert len(cliente.llamadas) == 1


def test_earthsearch_es_la_coleccion_1():
    parametros = stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"), catalogo="earthsearch")
    assert parametros["collections"] == ["sentinel-2-c1-l2a"]


def test_earthsearch_l2a_esta_desactivado():
    for llamada in (
        lambda: stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"), catalogo="earthsearch-l2a"),
        lambda: stac.conectar_catalogo("earthsearch-l2a"),
        lambda: stac.parametros_carga("sentinel-2", ["B04"], 30, "EPSG:32719", catalogo="earthsearch-l2a"),
    ):
        with pytest.raises(ValueError, match="desactivado") as error:
            llamada()
        assert "inconsistentes" in str(error.value)  # explica el motivo (metadatos de offset)


def test_catalogo_desconocido():
    with pytest.raises(ValueError, match="no reconocido") as error:
        stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"), catalogo="google")
    assert "planetary, earthsearch" in str(error.value) and "earthsearch-l2a" not in str(error.value)


def test_cambio_en_una_linea(monkeypatch):
    """Cambiar CATALOGO_SENTINEL2 cambia Sentinel-2, pero no Landsat."""
    monkeypatch.setattr(stac, "CATALOGO_SENTINEL2", "earthsearch")
    s2 = stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"))
    assert s2["collections"] == ["sentinel-2-c1-l2a"]
    landsat = stac.parametros_busqueda("landsat", BBOX, ("2024-01-01", "2024-01-31"))
    assert landsat["collections"] == ["landsat-c2-l2"]


def test_el_cliente_manda_sobre_el_valor_por_defecto(monkeypatch):
    monkeypatch.setattr(stac, "CATALOGO_SENTINEL2", "earthsearch")
    pc = ClienteFalso(stac.URL_PLANETARY)
    es = ClienteFalso(stac.URL_EARTHSEARCH)
    assert stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"), catalogo=pc)["collections"] == [
        "sentinel-2-l2a"
    ]
    assert stac.parametros_busqueda("sentinel-2", BBOX, ("2024-01-01", "2024-01-31"), catalogo=es)["collections"] == [
        "sentinel-2-c1-l2a"
    ]


def test_landsat_solo_en_planetary():
    parametros = stac.parametros_busqueda("landsat", BBOX, ("2024-01-01", "2024-01-31"), nubes_max=15)
    assert parametros["collections"] == ["landsat-c2-l2"]
    assert parametros["query"] == {"eo:cloud_cover": {"lt": 15}, "platform": {"in": ["landsat-8", "landsat-9"]}}
    stac.parametros_busqueda("landsat", BBOX, ("2024-01-01", "2024-01-31"), catalogo=ClienteFalso(stac.URL_PLANETARY))
    with pytest.raises(ValueError, match="Landsat solo está en Planetary Computer"):
        stac.parametros_busqueda("landsat", BBOX, ("2024-01-01", "2024-01-31"), catalogo=ClienteFalso(stac.URL_EARTHSEARCH))


# ---------------------------------------------------------------- hueco de Earth Search c1
@pytest.mark.parametrize(
    "inicio,fin,avisa",
    [
        ("2021-12-01", "2021-12-29", False),  # el 29-12 todavía tiene tomas
        ("2021-12-01", "2021-12-30", True),  # toca el primer día del hueco
        ("2022-03-01", "2022-03-31", True),
        ("2022-12-06", "2023-01-31", True),  # toca el último día del hueco
        ("2022-12-07", "2023-01-31", False),  # el 07-12 ya tiene tomas
        ("2021-01-01", "2023-12-31", True),  # lo cubre por completo
    ],
)
def test_aviso_hueco_c1(inicio, fin, avisa):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        stac.parametros_busqueda("sentinel-2", BBOX, (inicio, fin), catalogo="earthsearch")
    textos = [str(a.message) for a in avisos]
    assert any("casi no tiene imágenes entre 2021-12-29 y 2022-12-07" in t for t in textos) is avisa


def test_el_hueco_solo_afecta_a_earthsearch():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stac.parametros_busqueda("sentinel-2", BBOX, ("2022-03-01", "2022-03-31"), catalogo="planetary")
        stac.parametros_busqueda("landsat", BBOX, ("2022-03-01", "2022-03-31"))


# ---------------------------------------------------------------- conectar_catalogo
def test_conectar_catalogo_entrega_el_cliente_nativo(monkeypatch):
    abiertos = []

    def abrir(url, **kwargs):
        abiertos.append((url, kwargs))
        return ClienteFalso(url)

    monkeypatch.setattr(stac.Client, "open", abrir)
    assert isinstance(stac.conectar_catalogo(), ClienteFalso)  # el objeto de Client.open, sin envolver
    stac.conectar_catalogo("earthsearch")
    (url_pc, kw_pc), (url_es, kw_es) = abiertos
    assert url_pc == stac.URL_PLANETARY and kw_pc["modifier"] is stac.planetary_computer.sign_inplace
    assert url_es == stac.URL_EARTHSEARCH and kw_es == {}


# ---------------------------------------------------------------- preparar_items
def test_preparar_items_quita_duplicados_y_ordena(capsys):
    items = leer_items("pc_2021-12-14")
    assert len(items) == 2  # baseline 03.00 y 04.00 de la misma toma
    listos = stac.preparar_items(items)
    assert [stac.baseline_procesamiento(i) for i in listos] == ["04.00"]
    assert "Se encontraron 1 imágenes (sentinel-2-l2a, planetary) (1 versiones repetidas descartadas)" in (
        capsys.readouterr().out
    )
    desordenados = leer_items("pc_2025-02-16") + leer_items("pc_2020-12-04") + leer_items("pc_2022-01-31")
    assert [i.datetime.year for i in stac.preparar_items(desordenados, verbose=False)] == [2020, 2022, 2025]


def test_preparar_items_acepta_una_item_collection():
    coleccion = pystac.ItemCollection(leer_items("pc_2021-12-14"))
    listos = stac.preparar_items(coleccion, verbose=False)
    assert isinstance(listos, list) and len(listos) == 1


def test_preparar_items_conserva_misma_fecha_de_dos_satelites():
    assert len(stac.preparar_items(leer_items("pc_2026-06-06"), verbose=False)) == 2


def test_preparar_items_no_mezcla_catalogos():
    mezcla = leer_items("pc_2022-01-31") + leer_items("c1_2025-02-16")
    with pytest.raises(ValueError, match="varios catálogos"):
        stac.preparar_items(mezcla)


def test_preparar_items_rechaza_earthsearch_l2a():
    with pytest.raises(ValueError, match="desactivado"):
        stac.preparar_items(leer_items("es_2022-03-19"))


def test_preparar_items_sin_resultados(capsys):
    assert stac.preparar_items([]) == []
    assert "Prueba ampliando las fechas" in capsys.readouterr().out


def test_preparar_items_deja_pasar_landsat():
    landsat = leer_items("pc_2020-12-04")[0].clone()
    landsat.collection_id = "landsat-c2-l2"
    assert stac.preparar_items([landsat, landsat], verbose=False) == [landsat, landsat]
