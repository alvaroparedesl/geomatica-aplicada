"""
Qué demuestra este archivo (sin internet): la parte de CARGA de utils.stac, con
la llamada nativa `odc.stac.load(items, **parametros_carga(...))` de punta a punta.

Se usan ítems STAC reales (grabados) pero sus archivos se reemplazan por
GeoTIFF pequeños escritos en una carpeta temporal. Los DN de cada archivo se
calculan como los guardaría cada catálogo (con o sin el +1000), así que al
final la reflectancia debe ser la misma en todas las fechas y catálogos.
A los ítems que deberían descartarse por duplicados se les escriben DN
erróneos: si se colaran en la carga, la prueba fallaría.

Garantías que se prueban:

* `parametros_carga`: nombres de banda de cada catálogo, firma de Planetary Computer
  (`patch_url`), `groupby` (por día y offset en Sentinel-2) y rechazo de
  catálogos desactivados.
* `preparar_items`: las versiones duplicadas no llegan a la carga, y dos tiles de la misma
  toma con offsets distintos quedan en fechas distintas (1 ms) para poder asignar su offset.
* El `groupby` nunca junta en una fecha imágenes con offsets distintos.
* `agregar_offset`: un offset por fecha, y errores claros si los datos no
  corresponden a los ítems (duplicados, catálogos mezclados, carga sin `groupby`).
* `renombrar_bandas`: B02, B04, B08, SCL en todos los catálogos.
* Reflectancia final idéntica en Planetary Computer, Earth Search c1 y l2a.
"""

import numpy as np
import odc.stac
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from utils import raster, stac
from utils.raster import COORD_OFFSET

from .conftest import leer_items

REFL = {"B02": 0.012, "B04": 0.05, "B08": 0.30}
EPSG = "EPSG:32718"
N = 12  # píxeles por lado (10 m)
X0, Y0 = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True).transform(-73.10, -36.75)
X0, Y0 = round(X0, -1), round(Y0, -1)


def _bbox():
    a_lonlat = Transformer.from_crs(EPSG, "EPSG:4326", always_xy=True)
    (oeste, este), (norte, sur) = a_lonlat.transform([X0 + 20, X0 + N * 10 - 20], [Y0 - 20, Y0 - N * 10 + 20])
    return [oeste, sur, este, norte]


def _escribir(ruta, valor, dtype):
    perfil = {"driver": "GTiff", "width": N, "height": N, "count": 1, "dtype": dtype, "crs": EPSG,
              "transform": from_origin(X0, Y0, 10, 10), "nodata": 0}
    with rasterio.open(ruta, "w", **perfil) as destino:
        destino.write(np.full((1, N, N), valor, dtype=dtype))


def _localizar(items, carpeta, descartar=()):
    """Reemplaza los archivos de cada ítem por GeoTIFF locales con DN conocidos."""
    for item in items:
        offset = stac.offset_sentinel2(item)
        erroneo = item.id in descartar
        for banda, r in REFL.items():
            asset = banda if banda in item.assets else stac.BANDAS_EARTHSEARCH[banda]
            dn = round((r - offset) / 0.0001) + (2000 if erroneo else 0)
            ruta = carpeta / f"{item.id}_{asset}.tif"
            _escribir(ruta, dn, "uint16")
            item.assets[asset].href = ruta.as_uri()
        asset_scl = "SCL" if "SCL" in item.assets else "scl"
        ruta = carpeta / f"{item.id}_scl.tif"
        _escribir(ruta, 4, "uint8")
        item.assets[asset_scl].href = ruta.as_uri()
    return items


def _cargar(items, catalogo, bandas=("B02", "B04", "B08", "SCL"), **extra):
    """El flujo completo del curso con llamadas nativas; devuelve (datos en DN, reflectancia)."""
    items = stac.preparar_items(items, verbose=False)
    parametros = stac.parametros_carga("sentinel-2", list(bandas), resolucion=10, crs=EPSG, catalogo=catalogo,
                                       bbox=_bbox(), **extra)
    datos = odc.stac.load(items, **parametros)
    datos = stac.renombrar_bandas(datos, catalogo)
    datos = stac.agregar_offset(datos, items)
    return datos, raster.preparar_reflectancia(datos, "sentinel-2")


def _verificar_reflectancia(refl):
    for banda, r in REFL.items():
        valores = refl[banda].values
        assert np.isfinite(valores).all(), banda
        np.testing.assert_allclose(valores, r, atol=1e-6, err_msg=banda)





# ---------------------------------------------------------------- parametros_carga
def test_parametros_carga_planetary():
    p = stac.parametros_carga("sentinel-2", ["B04", "B08", "SCL"], 30, "EPSG:32719", catalogo="planetary",
                              bbox=[0, 0, 1, 1], chunks={"x": 512, "y": 512})
    assert p == {
        "bands": ["B04", "B08", "SCL"], "crs": "EPSG:32719", "resolution": 30, "bbox": [0, 0, 1, 1],
        "chunks": {"x": 512, "y": 512}, "groupby": stac.agrupar_por_dia_y_offset, "patch_url": stac.firmar_url,
    }


def test_parametros_carga_acepta_argumentos_extra_de_odc_stac():
    p = stac.parametros_carga("sentinel-2", ["B04"], 10, "EPSG:32719", fail_on_error=False, dtype="float32")
    assert p["fail_on_error"] is False and p["dtype"] == "float32"
    assert p["groupby"] is stac.agrupar_por_dia_y_offset  # lo demás sigue igual
    landsat = stac.parametros_carga("landsat", ["red"], 30, "EPSG:32719", fail_on_error=False, groupby="solar_day")
    assert landsat["fail_on_error"] is False and landsat["groupby"] == "solar_day"


def test_parametros_carga_sentinel2_no_deja_cambiar_el_groupby():
    with pytest.raises(ValueError, match="no se puede cambiar `groupby`"):
        stac.parametros_carga("sentinel-2", ["B04"], 10, "EPSG:32719", groupby="solar_day")


def test_cargar_ventanas_pasa_los_argumentos_extra_a_odc_stac(tmp_path, monkeypatch):
    from utils import series

    llamadas = []
    cargar_real = odc.stac.load

    def espia(items, **kwargs):
        llamadas.append(kwargs)
        return cargar_real(items, **kwargs)

    monkeypatch.setattr(odc.stac, "load", espia)
    items = _localizar(leer_items("pc_2022-01-31"), tmp_path)
    items = stac.preparar_items(items, verbose=False)
    oeste, sur, este, norte = _bbox()
    puntos = pd.DataFrame({"id": ["p1"], "nombre": ["uno"], "lat": [(sur + norte) / 2], "lon": [(oeste + este) / 2], "tipo": ["agua"]})
    ventanas = series.cargar_ventanas(items, puntos, bandas=["B04", "B08", "SCL"], radio_m=40, crs=EPSG, catalogo="planetary",
                                      verbose=False, fail_on_error=False)
    assert llamadas and all(l["fail_on_error"] is False for l in llamadas)
    assert "offset_reflectancia" in ventanas["p1"].coords


def test_parametros_carga_sin_opcionales():
    p = stac.parametros_carga("sentinel-2", ["B04"], 10, "EPSG:32719")  # catálogo por defecto: planetary
    assert set(p) == {"bands", "crs", "resolution", "groupby", "patch_url"}


def test_parametros_carga_earthsearch_usa_los_nombres_nativos():
    p = stac.parametros_carga("sentinel-2", ["B02", "B04", "B08", "B11", "SCL"], 10, "EPSG:32719",
                              catalogo="earthsearch")
    assert p["bands"] == ["blue", "red", "nir", "swir16", "scl"]
    assert "patch_url" not in p  # sin firma en AWS
    assert p["groupby"] is stac.agrupar_por_dia_y_offset


def test_parametros_carga_landsat():
    p = stac.parametros_carga("landsat", ["red", "nir08", "qa_pixel"], 30, "EPSG:32719")
    assert p["bands"] == ["red", "nir08", "qa_pixel"]
    assert p["groupby"] == "solar_day" and p["patch_url"] is stac.firmar_url
    with pytest.raises(ValueError, match="Landsat solo está en Planetary Computer"):
        stac.parametros_carga("landsat", ["red"], 30, "EPSG:32719", catalogo="earthsearch")


def test_parametros_carga_sigue_el_catalogo_por_defecto(monkeypatch):
    monkeypatch.setattr(stac, "CATALOGO_SENTINEL2", "earthsearch")
    assert stac.parametros_carga("sentinel-2", ["B04"], 10, "EPSG:32719")["bands"] == ["red"]
    assert stac.parametros_carga("landsat", ["red"], 30, "EPSG:32719")["bands"] == ["red"]  # Landsat sigue en PC


def test_firmar_url_renueva_el_token(monkeypatch):
    firmadas = []
    monkeypatch.setattr(stac.planetary_computer, "sign", lambda href: firmadas.append(href) or href + "?token-nuevo")
    viejo = "https://cuenta.blob.core.windows.net/contenedor/B04.tif?se=2020-01-01&sig=vencido"
    assert stac.firmar_url(viejo) == "https://cuenta.blob.core.windows.net/contenedor/B04.tif?token-nuevo"
    assert firmadas == ["https://cuenta.blob.core.windows.net/contenedor/B04.tif"]


def test_agrupar_por_dia_y_offset():
    s2b, s2a = leer_items("pc_2026-06-06")  # mismo día, dos satélites, mismo offset
    assert stac.agrupar_por_dia_y_offset(s2a) == stac.agrupar_por_dia_y_offset(s2b)
    assert stac.agrupar_por_dia_y_offset(s2a)[1] == -0.1
    s2b.properties["s2:processing_baseline"] = "03.00"
    assert stac.agrupar_por_dia_y_offset(s2b)[1] == 0.0
    assert stac.agrupar_por_dia_y_offset(s2b)[0] == stac.agrupar_por_dia_y_offset(s2a)[0]
    assert stac.agrupar_por_dia_y_offset(s2b, object(), 3) != stac.agrupar_por_dia_y_offset(s2a)  # firma de odc.stac


# ---------------------------------------------------------------- carga de punta a punta
def test_planetary(tmp_path):
    archivos = ["pc_2021-12-14", "pc_2022-01-23", "pc_2022-01-31", "pc_2025-02-16"]
    items = [i for a in archivos for i in leer_items(a)]
    descartar = {"S2A_MSIL2A_20211214T143731_R096_T18HXE_20211215T041443"}  # 03.00 duplicada
    _localizar(items, tmp_path, descartar)
    ds, refl = _cargar(items, "planetary")
    assert list(ds.data_vars) == ["B02", "B04", "B08", "SCL"]
    assert [str(t)[:10] for t in ds.time.values] == ["2021-12-14", "2022-01-23", "2022-01-31", "2025-02-16"]
    assert ds[COORD_OFFSET].values.tolist() == [-0.1, 0.0, -0.1, -0.1]
    assert ds[COORD_OFFSET].attrs["catalogo"] == "planetary"
    _verificar_reflectancia(refl)


def test_earthsearch_l2a_desactivado_pero_su_regla_sigue_probada(tmp_path, con_earthsearch_l2a):
    archivos = ["es_2020-12-04", "es_2022-01-31", "es_2022-03-19", "es_2024-11-11", "es_2025-03-18"]
    items = [i for a in archivos for i in leer_items(a)]
    descartar = {"S2B_18HXE_20201204_0_L2A", "S2B_18HXE_20220319_0_L2A"}
    _localizar(items, tmp_path, descartar)
    ds, refl = _cargar(items, "earthsearch-l2a")
    fechas = [str(t)[:10] for t in ds.time.values]
    # 2024-11-11: S2A y S2C (ambos ya restados) se combinan en una fecha
    assert fechas == ["2020-12-04", "2022-01-31", "2022-03-19", "2024-11-11", "2025-03-18"]
    assert ds[COORD_OFFSET].values.tolist() == [0.0, -0.1, 0.0, 0.0, 0.0]
    assert ds[COORD_OFFSET].attrs["catalogo"] == "earthsearch-l2a"
    _verificar_reflectancia(refl)


def test_earthsearch_c1(tmp_path):
    items = leer_items("c1_2021-12-14") + leer_items("c1_2025-02-16") + leer_items("c1_2024-11-11")
    _localizar(items, tmp_path)
    ds, refl = _cargar(items, "earthsearch")
    assert list(ds.data_vars) == ["B02", "B04", "B08", "SCL"]  # renombradas desde blue, red, nir, scl
    # 2024-11-11: S2A y S2C con el mismo offset se combinan en una sola fecha
    assert [str(t)[:10] for t in ds.time.values] == ["2021-12-14", "2024-11-11", "2025-02-16"]
    assert ds[COORD_OFFSET].values.tolist() == [-0.1, -0.1, -0.1]
    assert ds[COORD_OFFSET].attrs["catalogo"] == "earthsearch"
    _verificar_reflectancia(refl)


def test_earthsearch_sin_renombrar_tiene_los_nombres_nativos(tmp_path):
    items = _localizar(leer_items("c1_2025-02-16"), tmp_path)
    p = stac.parametros_carga("sentinel-2", ["B04", "SCL"], 10, EPSG, catalogo="earthsearch", bbox=_bbox())
    crudo = odc.stac.load(items, **p)
    assert list(crudo.data_vars) == ["red", "scl"]
    assert list(stac.renombrar_bandas(crudo, "earthsearch").data_vars) == ["B04", "SCL"]
    assert stac.renombrar_bandas(crudo, "earthsearch")["B04"].equals(crudo["red"])


def test_mismo_dia_con_offsets_distintos_quedan_separados(tmp_path):
    """Si un día mezcla imágenes con y sin +1000, quedan como dos fechas ordenadas."""
    items = leer_items("pc_2026-06-06")
    antiguo = [i for i in items if i.id.startswith("S2B")][0]
    antiguo.properties["s2:processing_baseline"] = "03.00"  # caso artificial
    _localizar(items, tmp_path)
    ds, refl = _cargar(items, "planetary")
    assert ds.sizes["time"] == 2
    assert ds.indexes["time"].is_monotonic_increasing
    # S2B (14:37, ahora 03.00) antes que S2A (14:47, 05.12)
    assert ds[COORD_OFFSET].values.tolist() == [0.0, -0.1]
    _verificar_reflectancia(refl)


def test_mismo_dia_dos_satelites_se_combinan(tmp_path):
    items = _localizar(leer_items("pc_2026-06-06"), tmp_path)
    ds, refl = _cargar(items, "planetary")
    assert ds.sizes["time"] == 1 and ds[COORD_OFFSET].values.tolist() == [-0.1]
    _verificar_reflectancia(refl)


def test_misma_toma_en_tres_catalogos_da_la_misma_reflectancia(tmp_path, con_earthsearch_l2a):
    resultados = []
    for archivo, catalogo in (("pc_2021-12-14", "planetary"), ("es_2021-12-14", "earthsearch-l2a"),
                              ("c1_2021-12-14", "earthsearch")):
        items = _localizar(leer_items(archivo), tmp_path)
        resultados.append(_cargar(items, catalogo)[1])
    for refl in resultados[1:]:
        for banda in REFL:
            np.testing.assert_allclose(refl[banda].values, resultados[0][banda].values, atol=1e-6)


def test_carga_perezosa(tmp_path):
    items = _localizar(leer_items("c1_2025-02-16") + leer_items("c1_2025-03-18"), tmp_path)
    ds, _ = _cargar(items, "earthsearch", bandas=("B04", "SCL"), chunks={"x": 6, "y": 6})
    assert ds["B04"].chunks is not None
    assert ds[COORD_OFFSET].values.tolist() == [-0.1, -0.1]
    refl = raster.preparar_reflectancia(ds, "sentinel-2").compute()
    np.testing.assert_allclose(refl["B04"].values, 0.05, atol=1e-6)


# ---------------------------------------------------------------- garantías y errores
def test_las_versiones_duplicadas_no_llegan_a_la_carga(tmp_path):
    """Con un DN erróneo en el duplicado 03.00, el resultado sería distinto si se colara."""
    items = leer_items("pc_2021-12-14")
    _localizar(items, tmp_path, {"S2A_MSIL2A_20211214T143731_R096_T18HXE_20211215T041443"})
    ds, refl = _cargar(items, "planetary")
    assert ds.sizes["time"] == 1 and ds[COORD_OFFSET].values.tolist() == [-0.1]
    _verificar_reflectancia(refl)


def test_agregar_offset_exige_items_sin_duplicados(tmp_path):
    items = leer_items("pc_2021-12-14")  # 03.00 y 04.00 de la misma toma
    _localizar(items, tmp_path)
    ds, _ = _cargar(items, "planetary")
    with pytest.raises(ValueError, match="preparar_items"):
        stac.agregar_offset(ds, items)


def test_no_mezcla_catalogos(tmp_path):
    mezcla = _localizar(leer_items("pc_2022-01-31") + leer_items("c1_2025-02-16"), tmp_path)
    with pytest.raises(ValueError, match="varios catálogos"):
        stac.preparar_items(mezcla)
    ds, _ = _cargar(leer_items("pc_2022-01-31"), "planetary")
    with pytest.raises(ValueError, match="varios catálogos"):  # tampoco se cuela por agregar_offset
        stac.agregar_offset(ds, mezcla)


def test_cargar_sin_el_groupby_del_curso_se_detecta(tmp_path):
    """Si se carga con groupby="solar_day" y hay offsets distintos en un día, agregar_offset avisa."""
    items = leer_items("pc_2026-06-06")
    items[0].properties["s2:processing_baseline"] = "03.00"
    _localizar(items, tmp_path)
    p = stac.parametros_carga("sentinel-2", ["B04"], 10, EPSG, bbox=_bbox())
    p["groupby"] = "solar_day"
    datos = odc.stac.load(items, **p)
    assert datos.sizes["time"] == 1  # se mezclaron
    with pytest.raises(ValueError, match="offsets de reflectancia distintos"):
        stac.agregar_offset(datos, items)


def test_agregar_offset_con_otros_items_o_fechas_ajenas(tmp_path):
    items = _localizar(leer_items("pc_2022-01-31"), tmp_path)
    datos, _ = _cargar(items, "planetary")
    ajenos = _localizar(leer_items("pc_2025-02-16"), tmp_path)
    with pytest.raises(ValueError, match="no corresponden a los ítems"):
        stac.agregar_offset(datos.drop_vars(COORD_OFFSET), ajenos)


def test_agregar_offset_sin_items(tmp_path):
    items = _localizar(leer_items("pc_2022-01-31"), tmp_path)
    datos, _ = _cargar(items, "planetary")
    with pytest.raises(ValueError, match="No hay imágenes"):
        stac.agregar_offset(datos, [])


def test_agregar_offset_en_landsat_no_cambia_nada():
    landsat = leer_items("pc_2020-12-04")[0].clone()
    landsat.collection_id = "landsat-c2-l2"
    datos = object()
    assert stac.agregar_offset(datos, [landsat]) is datos


def test_earthsearch_l2a_no_se_puede_cargar_sin_reactivarlo(tmp_path):
    items = _localizar(leer_items("es_2022-03-19"), tmp_path)
    with pytest.raises(ValueError, match="desactivado"):
        stac.preparar_items(items)


def test_tiles_de_la_misma_toma_con_offsets_distintos_se_cargan_con_su_offset(tmp_path):
    """El caso del 2022-01-20 (Conguillio): mismo instante, un tile con baseline 03.00 y otro con 04.00."""
    from utils import series

    base = leer_items("pc_2022-01-31")[0]
    original, reprocesado = base.clone(), base.clone()
    original.id += "_T19HBS"
    original.properties.update({"s2:mgrs_tile": "19HBS", "s2:processing_baseline": "03.00", "eo:cloud_cover": 10})
    reprocesado.id += "_T19HBT"
    reprocesado.properties.update({"s2:mgrs_tile": "19HBT", "s2:processing_baseline": "04.00", "eo:cloud_cover": 50})
    items = _localizar([original, reprocesado], tmp_path)
    datos, refl = _cargar(items, "planetary")
    assert datos.sizes["time"] == 2 and datos["time"].to_index().is_unique
    assert sorted(datos[COORD_OFFSET].values) == [-0.1, 0.0]
    _verificar_reflectancia(refl)  # cada tile con su offset: la misma reflectancia en ambos
    datos = series.agregar_nubes_escena(datos, stac.preparar_items(items, verbose=False))
    nubes = dict(zip(datos[COORD_OFFSET].values.tolist(), datos["nubes_escena"].values.tolist()))
    assert nubes == {0.0: 10.0, -0.1: 50.0}  # y cada fecha conserva las nubes de su grupo
