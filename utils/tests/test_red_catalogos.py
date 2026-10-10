"""
Qué demuestra este archivo (CON internet, marcadas `red`): que el flujo nativo del
curso (`catalogo.search` + `odc.stac.load` con los parámetros de utils.stac)
entrega la reflectancia correcta en los catálogos reales.

Carga la misma toma Sentinel-2 desde los distintos catálogos sobre Concepción /
Talcahuano (agua y tierra) y verifica que, después de aplicar el offset de cada
imagen, la reflectancia coincide. También compara con Landsat (referencia
independiente), revisa la continuidad antes y después de 2022 y prueba la
búsqueda y la carga con los clientes nativos.

Los tres nombres de catálogo de las pruebas son "planetary", "earthsearch"
(Colección 1) y "earthsearch-l2a", que está desactivado para los estudiantes y
aquí se reactiva (`con_earthsearch_l2a`) para seguir comprobando su regla de
offset sobre píxeles reales.

Ejecutar solo estas: `pytest utils -m red`. Omitirlas: `pytest utils -m "not red"`.
Son lentas (descargan imágenes) y necesitan acceso a Planetary Computer y AWS.
"""

import numpy as np
import odc.stac
import pytest

from utils import raster, stac

pytestmark = [pytest.mark.red, pytest.mark.usefixtures("con_earthsearch_l2a")]

MAIPO = [-70.8, -33.7, -70.6, -33.6]
EPSG_CONCE = "EPSG:32718"
# Áreas pequeñas (unos 2 km) para que la descarga sea rápida
OCEANO = [-73.27, -36.83, -73.24, -36.80]  # mar abierto al oeste de Hualpén
BAHIA = [-73.04, -36.68, -73.01, -36.66]  # Bahía de Concepción
TIERRA = [-73.16, -36.82, -73.11, -36.78]  # Hualpén (bosque, praderas, urbano)
BANDAS = ["B02", "B04", "B08", "SCL"]


def _buscar(catalogo, bbox, fechas, sensor="sentinel-2", nubes_max=100, tile=None):
    """Búsqueda nativa: catalogo.search(**parametros_busqueda(...)) + preparar_items."""
    nombre = catalogo if sensor == "sentinel-2" else "planetary"
    cliente = stac.conectar_catalogo(nombre)
    # el nombre (no el cliente) distingue "earthsearch" (c1) de "earthsearch-l2a": ambos usan la misma URL
    busqueda = cliente.search(**stac.parametros_busqueda(sensor, bbox, fechas, nubes_max, catalogo=nombre))
    items = stac.preparar_items(busqueda.item_collection(), verbose=False)
    return [i for i in items if tile is None or stac._tile(i).endswith(tile)]


def _cargar(items, bandas, bbox, catalogo, resolucion, crs=EPSG_CONCE, sensor="sentinel-2"):
    """Carga nativa: odc.stac.load(items, **parametros_carga(...)) + renombrar_bandas + agregar_offset."""
    parametros = stac.parametros_carga(sensor, bandas, resolucion, crs, catalogo=catalogo, bbox=bbox)
    datos = odc.stac.load(items, **parametros)
    if sensor == "sentinel-2":
        datos = stac.agregar_offset(stac.renombrar_bandas(datos, catalogo), items)
    return datos


def _cargar_fecha(catalogo, bbox, fecha, resolucion=20):
    items = _buscar(catalogo, bbox, (fecha, fecha), tile="18HXE")
    if not items:
        return None
    return _cargar(items, BANDAS, bbox, catalogo, resolucion).isel(time=0).compute()


def _refl(ds):
    return raster.convertir_a_reflectancia(ds.drop_vars("SCL"), "sentinel-2")


# (fecha, catálogos con esa toma, caso)
TOMAS = [
    ("2020-12-04", ["planetary", "earthsearch-l2a", "earthsearch"], "antes de 2022 (02.12 / 05.00 ya restado / c1)"),
    ("2021-12-14", ["planetary", "earthsearch-l2a", "earthsearch"], "dic-2021 reprocesada (PC 04.00 / ES 05.00 restado)"),
    ("2022-01-31", ["planetary", "earthsearch-l2a"], "04.00, Earth Search l2a sin restar (ventana feb-2022)"),
    ("2022-03-07", ["planetary", "earthsearch-l2a"], "04.00, Earth Search l2a dice False pero ya restó"),
    ("2022-03-22", ["planetary", "earthsearch-l2a", "earthsearch"], "05.10 en los tres"),
    ("2025-02-16", ["planetary", "earthsearch-l2a", "earthsearch"], "05.11, Earth Search l2a dice False pero ya restó"),
    ("2025-03-18", ["planetary", "earthsearch-l2a", "earthsearch"], "05.11, Earth Search l2a ya restado"),
]


@pytest.mark.parametrize("fecha,catalogos,caso", TOMAS, ids=[t[0] for t in TOMAS])
@pytest.mark.parametrize("area,bbox", [("oceano", OCEANO), ("tierra", TIERRA)])
def test_misma_toma_misma_reflectancia(fecha, catalogos, caso, area, bbox):
    datos = {c: _cargar_fecha(c, bbox, fecha) for c in catalogos}
    assert all(d is not None for d in datos.values()), f"falta la toma {fecha} en algún catálogo"
    ref = datos["planetary"]
    scl = ref["SCL"].values
    mascara = np.isin(scl, [6] if area == "oceano" else [4, 5])
    assert mascara.mean() > 0.3, f"pocos píxeles útiles en {area}"
    refl_ref = _refl(ref)
    for catalogo, ds in datos.items():
        refl = _refl(ds)
        for banda in ("B02", "B04", "B08"):
            a = refl_ref[banda].values[mascara]
            b = refl[banda].values[mascara]
            ok = np.isfinite(a) & np.isfinite(b)
            assert ok.mean() > 0.8, f"{catalogo} {banda}: demasiados NaN"
            # sesgo: un error de offset da 0.1; las vistas reducidas de cada
            # catálogo agregan ruido píxel a píxel en tierra (NIR ~0.01)
            sesgo = np.median(b[ok] - a[ok])
            assert abs(sesgo) < 0.005, f"{catalogo} vs planetary {banda} {area}: sesgo {sesgo:+.4f} ({caso})"
            ruido = np.median(np.abs(b[ok] - a[ok]))
            assert ruido < (0.005 if area == "oceano" else 0.025), f"{catalogo} {banda} {area}: |dif| {ruido:.4f}"
        if area == "oceano":
            # el agua es oscura: un error de offset la dejaría negativa (NaN) o en ~0.1
            rojo = refl["B04"].values[mascara]
            assert np.isfinite(rojo).mean() > 0.9
            assert 0 <= np.nanmedian(rojo) < 0.05


@pytest.mark.parametrize("catalogo", ["planetary", "earthsearch-l2a", "earthsearch"])
def test_agua_de_la_bahia_no_negativa(catalogo):
    ds = _cargar_fecha(catalogo, BAHIA, "2025-02-16")
    refl = _refl(ds)
    agua = ds["SCL"].values == 6
    for banda in ("B02", "B04", "B08"):
        valores = refl[banda].values[agua]
        assert np.isfinite(valores).mean() > 0.9, banda
        assert 0 <= np.nanmedian(valores) < 0.06, banda


@pytest.mark.parametrize("catalogo", ["planetary", "earthsearch-l2a"])
def test_continuidad_antes_y_despues_de_2022(catalogo):
    """Reflectancia del mar abierto en enero 2022: 03.00 (23-ene) vs 04.00 (31-ene)."""
    medianas = []
    for fecha in ("2022-01-23", "2022-01-31"):
        ds = _cargar_fecha(catalogo, OCEANO, fecha)
        refl = _refl(ds)
        agua = ds["SCL"].values == 6
        medianas.append(np.nanmedian(refl["B04"].values[agua]))
    assert all(0 <= m < 0.05 for m in medianas)
    assert abs(medianas[0] - medianas[1]) < 0.02  # sin offset correcto el salto sería ~0.1


def _serie_oceano(catalogo):
    items = _buscar(catalogo, OCEANO, ("2021-12-01", "2022-03-31"), nubes_max=30, tile="18HXE")
    if not items:
        return None
    ds = _cargar(items, ["B04", "SCL"], OCEANO, catalogo, 60)
    refl = raster.preparar_reflectancia(ds, "sentinel-2")
    serie = refl["B04"].median(["x", "y"]).to_series().dropna()
    serie.index = serie.index.normalize()
    return serie


def test_serie_que_cruza_el_cambio_de_baseline():
    """
    Mar abierto, dic-2021 a mar-2022 (03.00, 04.00 reprocesadas, ventana de
    febrero en Earth Search l2a, 05.00/05.10): misma serie en todos los catálogos.
    (Earth Search c1 tiene un hueco entre el 29-12-2021 y el 07-12-2022.)
    """
    series = {c: _serie_oceano(c) for c in ("planetary", "earthsearch-l2a", "earthsearch")}
    pc = series["planetary"]
    assert len(pc) > 5
    assert (pc >= 0).all() and (pc < 0.09).all()  # con un offset equivocado: ~0.1 o NaN
    for catalogo in ("earthsearch-l2a", "earthsearch"):
        otra = series[catalogo]
        if otra is None:
            continue  # c1 no tiene enero-marzo de 2022 (hueco)
        comunes = pc.index.intersection(otra.index)
        assert len(comunes) > 0
        dif = (otra[comunes] - pc[comunes]).abs()
        assert (dif < 0.005).all(), f"{catalogo}: {dif[dif >= 0.005].to_dict()}"


# ---------------------------------------------------------------- Landsat como referencia
PARES_LANDSAT = [
    # (fecha, caso) en el valle del Maipo: Landsat y Sentinel-2 el mismo día
    ("2021-12-14", "S2 reprocesada 04.00 (PC) / 05.00 ya restado (ES)"),
    ("2025-02-16", "S2C 05.11"),
]
MAIPO_AGRO = [-70.80, -33.80, -70.70, -33.72]


def _ndvi_landsat(fecha):
    items = _buscar("planetary", MAIPO_AGRO, (fecha, fecha), sensor="landsat", nubes_max=20)
    ds = _cargar(items, ["red", "nir08", "qa_pixel"], MAIPO_AGRO, "planetary", 30, crs="EPSG:32719", sensor="landsat")
    refl = raster.preparar_reflectancia(ds, "landsat").isel(time=0)
    return refl, raster.calcular_ndvi(refl, "landsat")


@pytest.mark.parametrize("fecha,caso", PARES_LANDSAT, ids=[p[0] for p in PARES_LANDSAT])
@pytest.mark.parametrize("catalogo", ["planetary", "earthsearch-l2a", "earthsearch"])
def test_ndvi_coincide_con_landsat(fecha, caso, catalogo):
    refl_ls, ndvi_ls = _ndvi_landsat(fecha)
    items = _buscar(catalogo, MAIPO_AGRO, (fecha, fecha), nubes_max=20, tile="19HCC")
    assert items, f"no hay Sentinel-2 {fecha} en {catalogo}"
    ds = _cargar(items, ["B04", "B08", "SCL"], MAIPO_AGRO, catalogo, 30, crs="EPSG:32719")
    refl_s2 = raster.preparar_reflectancia(ds, "sentinel-2").isel(time=0)
    ndvi_s2 = raster.calcular_ndvi(refl_s2, "sentinel-2")
    ok = np.isfinite(ndvi_s2.values) & np.isfinite(ndvi_ls.values)
    assert ok.sum() > 1000
    dif_ndvi = np.mean(ndvi_s2.values[ok]) - np.mean(ndvi_ls.values[ok])
    dif_rojo = np.median(refl_s2["B04"].values[ok] - refl_ls["red"].values[ok])
    # con un offset equivocado: NDVI ~0.17 más bajo o ~0.4 más alto; rojo ~±0.1
    assert abs(dif_ndvi) < 0.05, f"NDVI S2 - Landsat = {dif_ndvi:+.3f} ({caso})"
    assert abs(dif_rojo) < 0.02, f"rojo S2 - Landsat = {dif_rojo:+.4f} ({caso})"


# ---------------------------------------------------------------- flujo nativo completo (Maipo)
@pytest.mark.parametrize("sensor,res", [("landsat", 30), ("sentinel-2", 60)])
@pytest.mark.parametrize("catalogo", ["planetary", "earthsearch"])
def test_flujo_nativo_busqueda_y_carga(sensor, res, catalogo):
    if sensor == "landsat" and catalogo != "planetary":
        pytest.skip("Landsat solo está en Planetary Computer")
    cliente = stac.conectar_catalogo(catalogo)
    parametros = stac.parametros_busqueda(sensor, MAIPO, ("2024-01-01", "2024-02-28"), nubes_max=15, catalogo=cliente)
    items = stac.preparar_items(cliente.search(**parametros).item_collection())
    assert len(items) > 0
    tabla = stac.resumir_imagenes(items)
    assert list(tabla.columns)[:3] == ["fecha", "nubes_%", "id"] and len(tabla) == len(items)
    bandas = raster.nombres_bandas(sensor, ["red", "green", "blue", "nir", "swir1"], con_calidad=True)
    ds = odc.stac.load(items, **stac.parametros_carga(sensor, bandas, res, "EPSG:32719", catalogo=cliente, bbox=MAIPO))
    if sensor == "sentinel-2":
        ds = stac.agregar_offset(stac.renombrar_bandas(ds, cliente), items)
    calidad = raster.SENSORES[sensor]["banda_calidad"]
    assert calidad in ds and ds.sizes["time"] > 0
    refl = raster.preparar_reflectancia(ds, sensor)
    ndvi = raster.calcular_ndvi(refl, sensor)
    assert -1 <= float(ndvi.min()) and float(ndvi.max()) <= 1
    assert 0.0 < float(refl[raster.SENSORES[sensor]["bandas"]["nir"]].mean()) < 0.6


def test_busqueda_vacia_y_utilidades():
    cliente = stac.conectar_catalogo("planetary")
    parametros = stac.parametros_busqueda("landsat", MAIPO, ("1990-01-01", "1990-01-02"), catalogo=cliente)
    assert stac.preparar_items(cliente.search(**parametros).item_collection(), verbose=False) == []
    items = _buscar("planetary", MAIPO, ("2024-01-01", "2024-02-28"), sensor="landsat", nubes_max=15)
    assert stac.escala_y_offset(items[0], "red") == (2.75e-05, -0.2)
    dem = stac.cargar_dem(MAIPO, resolucion=60)
    assert dem.ndim == 2 and 300 < float(dem.mean()) < 4000


def test_conectar_entrega_clientes_nativos():
    """`conectar_catalogo` devuelve el Client de pystac_client, y los parámetros respetan su catálogo."""
    from pystac_client import Client

    pc = stac.conectar_catalogo()
    es = stac.conectar_catalogo("earthsearch")
    assert isinstance(pc, Client) and isinstance(es, Client)
    for cliente, catalogo in ((pc, "planetary"), (es, "earthsearch")):
        parametros = stac.parametros_busqueda("sentinel-2", MAIPO, ("2024-01-01", "2024-01-31"), 30, catalogo=cliente)
        items = cliente.search(**parametros).item_collection()
        assert len(items) > 0 and all(stac.catalogo_de_item(i) == catalogo for i in items)
    with pytest.raises(ValueError, match="Landsat solo está en Planetary Computer"):
        stac.parametros_busqueda("landsat", MAIPO, ("2024-01-01", "2024-01-31"), catalogo=es)


def test_busqueda_sin_duplicados_en_transicion():
    """Dic-2021 en Planetary Computer tiene 03.00 y 04.00 de la misma toma."""
    cliente = stac.conectar_catalogo("planetary")
    parametros = stac.parametros_busqueda("sentinel-2", OCEANO, ("2021-12-14", "2021-12-14"), 100, catalogo=cliente)
    todos = list(cliente.search(**parametros).item_collection())
    limpios = stac.preparar_items(todos, verbose=False)
    assert len(todos) > len(limpios)
    assert all(stac.baseline_procesamiento(i) == "04.00" for i in limpios)


def test_hueco_de_earthsearch_c1():
    """
    Earth Search c1 casi no tiene imágenes entre el 29-12-2021 y el 07-12-2022 (por eso el aviso):
    en once meses solo hay unas pocas escenas sueltas, y en períodos normales hay varias por mes.
    """
    cliente = stac.conectar_catalogo("earthsearch")

    def cuantas(inicio, fin):
        parametros = stac.parametros_busqueda("sentinel-2", MAIPO, (inicio, fin), 100, catalogo=cliente)
        return len(cliente.search(**parametros).item_collection())

    with pytest.warns(UserWarning, match="casi no tiene imágenes entre"):
        en_el_hueco = cuantas("2022-01-01", "2022-12-01")
    assert en_el_hueco <= 3
    assert cuantas("2021-11-01", "2021-12-28") > 3  # antes del hueco
    assert cuantas("2022-12-08", "2022-12-31") > 3  # después del hueco
