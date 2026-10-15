"""
Qué demuestra este archivo (sin internet): el código de reflectancia que los cuadernos del taller
escriben a la vista de las y los estudiantes hace lo mismo que las funciones probadas de `utils`.

En los cuadernos 02 a 04 el offset de reflectancia de Sentinel-2 y la conversión de DN a reflectancia
no vienen de `utils`: se escriben en el cuaderno para que se lea cada paso. Como un offset mal
aplicado no da error (da una serie desplazada en 0.1), este archivo extrae ese código de los `.py`
y lo compara con `utils.stac.offset_sentinel2`, `utils.stac.agregar_offset` y
`utils.raster.convertir_a_reflectancia`:

* La regla del cuaderno 03 (`offset = -0.1 si baseline >= 4.0, si no 0`) da el mismo offset que
  `offset_sentinel2` en todos los ítems reales grabados de Planetary Computer y Earth Search c1.
* Las funciones `agregar_offset` y `reflectancia_sentinel2` del cuaderno 04 dan las mismas fechas,
  offsets y reflectancias que las de `utils`, con carga real de `odc.stac` sobre archivos locales
  (varias fechas, dos satélites el mismo día y versiones duplicadas de una toma), solo con Planetary Computer.
* El DN 0 (sin datos) y los valores fuera de 0 a 1 pasan a NaN igual que en `utils`.
* El cuaderno `analisis_completo` (un solo cuaderno, sin `utils`) repite esas reglas con sus propias funciones
  (`offset_item`, `grupo_dia_offset`, `quitar_duplicados`, `agregar_metadatos`, `reflectancia_sentinel2`): se
  comprueba que dan lo mismo que `utils`, incluidas las nubes de escena de cada fecha.
"""

import ast
import re
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import odc.stac
import pytest
import xarray as xr

from utils import raster, stac
from utils.raster import COORD_OFFSET
from utils.series import agregar_nubes_escena

from .conftest import FIXTURES, leer_items
from .test_carga_offline import EPSG, REFL, _bbox, _localizar
from .test_reflectancia import cubo_dn

TALLER = Path(__file__).resolve().parents[2] / "talleres" / "fair-series-temporales"


def _funciones_del_cuaderno(archivo, nombres, **importado):
    """Compila solo las funciones `nombres` escritas en un cuaderno (`.py` de jupytext) y las devuelve."""
    fuente = (TALLER / archivo).read_text(encoding="utf-8")
    definiciones = [n for n in ast.parse(fuente).body if isinstance(n, ast.FunctionDef) and n.name in nombres]
    assert {d.name for d in definiciones} == set(nombres), f"{archivo} no define {nombres}"
    espacio = {"np": np, **importado}  # lo único que los cuadernos importan para estas funciones
    exec(compile(ast.Module(body=definiciones, type_ignores=[]), archivo, "exec"), espacio)
    return types.SimpleNamespace(**{n: espacio[n] for n in nombres})


@pytest.fixture(scope="module")
def cuaderno04():
    return _funciones_del_cuaderno("04_series_temporales.py", ("agregar_offset", "reflectancia_sentinel2"))


# ---------------------------------------------------------------- cuaderno 03: la regla, escrita a mano
def _regla_del_cuaderno03():
    """Las dos líneas que calculan `offset` en el cuaderno 03, listas para ejecutar sobre un `item`."""
    fuente = (TALLER / "03_acceso_sentinel2.py").read_text(encoding="utf-8")
    coincidencia = re.search(r'^baseline = float\(item\.properties\["s2:processing_baseline"\]\).*\noffset = .*$', fuente, re.M)
    assert coincidencia, "no encuentro el cálculo del offset en el cuaderno 03"
    return compile(coincidencia.group(0), "03_acceso_sentinel2.py", "exec")


def _items_de_catalogos_activos():
    """Ítems reales grabados de Planetary Computer y Earth Search c1 (los catálogos que usa el curso)."""
    archivos = sorted(p.stem for p in (FIXTURES / "items").glob("*.json") if p.stem.startswith(("pc_", "c1_")))
    return [(a, item) for a in archivos for item in leer_items(a)]


@pytest.mark.parametrize("archivo,item", _items_de_catalogos_activos(), ids=lambda v: v if isinstance(v, str) else v.id[:30])
def test_la_regla_del_cuaderno03_coincide_con_offset_sentinel2(archivo, item):
    espacio = {"item": item}
    exec(_regla_del_cuaderno03(), espacio)
    assert espacio["offset"] == stac.offset_sentinel2(item), f"{archivo}: {item.id}"


# ---------------------------------------------------------------- cuaderno 04: las funciones
ESCENARIOS = {
    "varias fechas, antes y después de 2022": ["pc_2020-12-04", "pc_2022-01-23", "pc_2022-01-31", "pc_2025-02-16"],
    "dos satélites el mismo día": ["pc_2026-06-06"],
    "versión duplicada de una toma": ["pc_2021-12-14"],
}


def _cargar_sin_ayudas(items):
    """Carga nativa con `parametros_carga` y sin renombrar ni agregar el offset: lo hace cada cual."""
    parametros = stac.parametros_carga("sentinel-2", ["B02", "B04", "B08", "SCL"], resolucion=10, crs=EPSG, bbox=_bbox())
    return odc.stac.load(items, **parametros)


@pytest.mark.parametrize("nombre", ESCENARIOS)
def test_agregar_offset_y_reflectancia_del_cuaderno04_igualan_a_utils(nombre, cuaderno04, tmp_path):
    archivos = ESCENARIOS[nombre]
    items = [i for a in archivos for i in leer_items(a)]
    _localizar(items, tmp_path)
    items = stac.preparar_items(items, verbose=False)
    datos = _cargar_sin_ayudas(items)

    del_cuaderno = cuaderno04.agregar_offset(datos, items)
    de_utils = stac.agregar_offset(datos, items)
    assert del_cuaderno[COORD_OFFSET].dims == ("time",)
    np.testing.assert_array_equal(del_cuaderno[COORD_OFFSET].values, de_utils[COORD_OFFSET].values)

    espectrales = ["B02", "B04", "B08"]
    refl_cuaderno = cuaderno04.reflectancia_sentinel2(del_cuaderno[espectrales])
    refl_utils = raster.convertir_a_reflectancia(de_utils[espectrales], "sentinel-2")
    xr.testing.assert_allclose(refl_cuaderno, refl_utils)
    assert COORD_OFFSET not in refl_cuaderno.coords
    for banda, esperado in REFL.items():
        if banda in espectrales:
            np.testing.assert_allclose(refl_cuaderno[banda].values, esperado, atol=1e-6, err_msg=banda)


def test_reflectancia_del_cuaderno04_trata_igual_dn_cero_y_fuera_de_rango(cuaderno04):
    cubo = cubo_dn().drop_vars("SCL")
    cubo["B04"][0, 0, 0] = 0  # sin dato
    cubo["B02"][1, 1, 1] = 65000  # reflectancia muy por encima de 1
    cubo["B08"][3, 2, 2] = 100  # con offset -0.1 queda negativa
    del_cuaderno = cuaderno04.reflectancia_sentinel2(cubo)
    de_utils = raster.convertir_a_reflectancia(cubo, "sentinel-2")
    xr.testing.assert_identical(del_cuaderno, de_utils)
    assert np.isnan(del_cuaderno["B04"][0, 0, 0]) and np.isnan(del_cuaderno["B02"][1, 1, 1]) and np.isnan(del_cuaderno["B08"][3, 2, 2])


def test_reflectancia_del_cuaderno04_funciona_con_una_sola_fecha(cuaderno04):
    una = cubo_dn().drop_vars("SCL").isel(time=2)  # el offset queda como coordenada escalar
    xr.testing.assert_identical(cuaderno04.reflectancia_sentinel2(una),
                                raster.convertir_a_reflectancia(una, "sentinel-2"))


# ---------------------------------------------------------------- cuaderno analisis_completo: el mismo flujo, sin utils
FUNCIONES_COMPLETO = ("offset_item", "grupo_dia_offset", "quitar_duplicados", "desempatar_fechas", "agregar_metadatos",
                      "reflectancia_sentinel2")


@pytest.fixture(scope="module")
def completo():
    return _funciones_del_cuaderno("analisis_completo.py", FUNCIONES_COMPLETO, re=re, timedelta=timedelta)


def _items_planetary():
    return [(a, item) for a, item in _items_de_catalogos_activos() if a.startswith("pc_")]  # el cuaderno es solo Planetary Computer


@pytest.mark.parametrize("archivo,item", _items_planetary(), ids=lambda v: v if isinstance(v, str) else v.id[:30])
def test_offset_item_de_analisis_completo_coincide_con_offset_sentinel2(archivo, item, completo):
    assert completo.offset_item(item) == stac.offset_sentinel2(item), f"{archivo}: {item.id}"
    assert completo.grupo_dia_offset(item) == stac.agrupar_por_dia_y_offset(item)


@pytest.mark.parametrize("nombre", ESCENARIOS)
def test_analisis_completo_deja_las_mismas_imagenes_fechas_y_nubes_que_utils(nombre, completo, tmp_path):
    items = [i for a in ESCENARIOS[nombre] for i in leer_items(a)]
    _localizar(items, tmp_path)
    assert [i.id for i in completo.quitar_duplicados(items)] == [i.id for i in stac.preparar_items(items, verbose=False)]
    items = stac.preparar_items(items, verbose=False)

    parametros = stac.parametros_carga("sentinel-2", ["B02", "B04", "B08", "SCL"], resolucion=10, crs=EPSG, bbox=_bbox())
    datos = odc.stac.load(items, **parametros)
    cargado_con_el_cuaderno = odc.stac.load(items, **{**parametros, "groupby": completo.grupo_dia_offset})
    np.testing.assert_array_equal(cargado_con_el_cuaderno["time"].values, datos["time"].values)  # misma agrupación en fechas

    del_cuaderno = completo.agregar_metadatos(cargado_con_el_cuaderno, items)
    de_utils = agregar_nubes_escena(stac.agregar_offset(datos, items), items)
    np.testing.assert_array_equal(del_cuaderno[COORD_OFFSET].values, de_utils[COORD_OFFSET].values)
    np.testing.assert_allclose(del_cuaderno["nubes_escena"].values, de_utils["nubes_escena"].values)

    espectrales = ["B02", "B04", "B08"]
    xr.testing.assert_allclose(completo.reflectancia_sentinel2(del_cuaderno[espectrales]),
                               raster.convertir_a_reflectancia(de_utils[espectrales], "sentinel-2"))


def test_analisis_completo_asigna_el_offset_de_dos_tiles_con_la_misma_hora_igual_que_utils(completo, tmp_path):
    """El caso del 2022-01-20 (Conguillio): un tile reprocesado (04.00) y otro no (03.00) con la misma hora exacta."""
    base = leer_items("pc_2022-01-31")[0]
    original, reprocesado = base.clone(), base.clone()
    original.id += "_T19HBS"
    original.properties.update({"s2:mgrs_tile": "19HBS", "s2:processing_baseline": "03.00", "eo:cloud_cover": 10})
    reprocesado.id += "_T19HBT"
    reprocesado.properties.update({"s2:mgrs_tile": "19HBT", "s2:processing_baseline": "04.00", "eo:cloud_cover": 50})
    items = _localizar([original, reprocesado], tmp_path)

    del_cuaderno = completo.desempatar_fechas(completo.quitar_duplicados(items))
    de_utils = stac.preparar_items(items, verbose=False)
    assert [(i.id, i.datetime) for i in del_cuaderno] == [(i.id, i.datetime) for i in de_utils]
    assert original.datetime == reprocesado.datetime  # los ítems de entrada no cambian

    parametros = stac.parametros_carga("sentinel-2", ["B02", "B04", "B08", "SCL"], resolucion=10, crs=EPSG, bbox=_bbox())
    datos = odc.stac.load(de_utils, **parametros)
    cargado = odc.stac.load(del_cuaderno, **{**parametros, "groupby": completo.grupo_dia_offset})
    con_metadatos = completo.agregar_metadatos(cargado, del_cuaderno)
    esperado = agregar_nubes_escena(stac.agregar_offset(datos, de_utils), de_utils)
    assert cargado["time"].to_index().is_unique
    np.testing.assert_array_equal(con_metadatos[COORD_OFFSET].values, esperado[COORD_OFFSET].values)
    np.testing.assert_allclose(con_metadatos["nubes_escena"].values, esperado["nubes_escena"].values)
    assert sorted(con_metadatos[COORD_OFFSET].values) == [-0.1, 0.0]

    with pytest.raises(ValueError, match="desempatar_fechas"):  # sin el desempate el cuaderno avisa en vez de dar un offset falso
        completo.agregar_metadatos(odc.stac.load(items, **{**parametros, "groupby": completo.grupo_dia_offset}), items)
