"""
Qué demuestra este archivo: pruebas generales de utils/ (vectorial, raster, gráficos)
con datos sintéticos; las del offset de Sentinel-2 están en los demás archivos.

Ejecutar desde la raíz del repo: `pytest utils -v`. Las marcadas con `red`
usan internet; para omitirlas: `pytest utils -m "not red"`.
"""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon

from utils import raster, viz
from utils.vector import calcular_area

BBOX = [-70.8, -33.7, -70.6, -33.6]


def _dataset_sintetico(sensor="landsat", ny=20, nx=30, nt=3):
    """Dataset en DN con la misma estructura que entrega odc.stac."""
    config = raster.SENSORES[sensor]
    rng = np.random.default_rng(0)
    coords = {
        "time": pd.date_range("2024-01-01", periods=nt, freq="10D"),
        "y": np.linspace(6280000, 6270000, ny),
        "x": np.linspace(330000, 350000, nx),
    }
    base = 10000 if sensor == "landsat" else 2000
    datos = {}
    for nombre in config["bandas"].values():
        datos[nombre] = (("time", "y", "x"), rng.integers(base, base * 2, (nt, ny, nx)).astype("uint16"))
    ds = xr.Dataset(datos, coords=coords)
    if sensor == "sentinel-2":  # como lo deja utils.stac.agregar_offset
        ds = ds.assign_coords({raster.COORD_OFFSET: ("time", np.full(nt, -0.1))})
    return ds.rio.write_crs("EPSG:32719")


# ---------------------------------------------------------------- vector
def test_calcular_area():
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    km2 = calcular_area(gdf, "km2")
    assert 12000 < km2.iloc[0] < 12500  # ~1 grado x 1 grado en el ecuador
    assert calcular_area(gdf, "ha").iloc[0] == pytest.approx(km2.iloc[0] * 100)
    with pytest.raises(ValueError):
        calcular_area(gdf, "acres")


@pytest.mark.red
def test_descargar_natural_earth():
    from utils.vector import descargar_natural_earth

    paises = descargar_natural_earth()
    assert len(paises) > 150 and "NAME" in paises
    assert "Chile" in paises["NAME"].values
    assert descargar_natural_earth().equals(paises)  # segunda vez: desde caché


# ---------------------------------------------------------------- raster
def test_configuracion_y_nombres():
    assert raster.obtener_configuracion_sensor("landsat")["coleccion"] == "landsat-c2-l2"
    assert raster.obtener_configuracion_sensor(raster.SENSORES["landsat"]) is raster.SENSORES["landsat"]
    with pytest.raises(ValueError):
        raster.obtener_configuracion_sensor("modis")
    assert raster.nombres_bandas("landsat", ("nir", "red")) == ["nir08", "red"]
    assert raster.nombres_bandas("sentinel-2", ("nir",), con_calidad=True) == ["B08", "SCL"]


@pytest.mark.parametrize("sensor", ["landsat", "sentinel-2"])
def test_reflectancia_e_indices(sensor):
    ds = _dataset_sintetico(sensor)
    refl = raster.convertir_a_reflectancia(ds, sensor)
    assert float(refl.to_array().max()) <= 1
    indices = raster.calcular_indices(refl, sensor)
    assert list(indices.data_vars) == ["NDVI", "EVI", "NDWI"]
    ndvi = raster.calcular_ndvi(refl, sensor)
    b = raster.SENSORES[sensor]["bandas"]
    esperado = (refl[b["nir"]] - refl[b["red"]]) / (refl[b["nir"]] + refl[b["red"]])
    np.testing.assert_allclose(ndvi.values, esperado.where((esperado >= -1) & (esperado <= 1)).values)
    assert ndvi.name == "NDVI"


def test_division_por_cero_da_nan():
    ds = xr.Dataset({"nir08": ("x", [0.0, 0.5]), "red": ("x", [0.0, 0.1])})
    ndvi = raster.calcular_ndvi(ds, "landsat")
    assert np.isnan(ndvi.values[0]) and ndvi.values[1] == pytest.approx(0.4 / 0.6)


def test_mascara_landsat():
    # bit 3 = nube (8), bit 4 = sombra (16), bit 5 = nieve (32), 0 = limpio
    qa = xr.DataArray(np.array([0, 8, 16, 32, 64], dtype="uint16"), dims="x")
    assert raster.mascara_calidad_landsat(qa).values.tolist() == [True, False, False, True, True]
    assert raster.mascara_calidad_landsat(qa, excluir=["nieve"]).values.tolist() == [True, True, True, False, True]
    assert raster.mascara_calidad_landsat(qa, excluir=[3]).values.tolist() == [True, False, True, True, True]


def test_mascara_sentinel2():
    scl = xr.DataArray(np.array([4, 5, 8, 9, 3, 6], dtype="float32"), dims="x")
    assert raster.mascara_calidad_sentinel2(scl).values.tolist() == [True, True, False, False, False, True]


@pytest.mark.parametrize("sensor,calidad", [("landsat", "qa_pixel"), ("sentinel-2", "SCL")])
def test_preparar_reflectancia_enmascara(sensor, calidad):
    ds = _dataset_sintetico(sensor)
    nube = 8 if sensor == "landsat" else 9
    limpio = 0 if sensor == "landsat" else 4  # en SCL, 4 = vegetación (el 0 es "sin datos")
    ds[calidad] = xr.full_like(ds[raster.SENSORES[sensor]["bandas"]["red"]], limpio, dtype="uint16")
    ds[calidad][0, 0, 0] = nube
    refl = raster.preparar_reflectancia(ds, sensor)
    assert calidad not in refl
    rojo = raster.SENSORES[sensor]["bandas"]["red"]
    assert np.isnan(refl[rojo][0, 0, 0]) and not np.isnan(refl[rojo][0, 0, 1])
    with pytest.raises(ValueError):
        raster.enmascarar_nubes(ds.drop_vars(calidad), sensor)


def test_series_puntos_y_estadisticas():
    refl = raster.convertir_a_reflectancia(_dataset_sintetico(), "landsat")
    indices = raster.calcular_indices(refl, "landsat", ("NDVI", "EVI"))
    puntos = pd.DataFrame(
        {"punto_id": ["a", "b"], "x": [335000, 345000], "y": [6279000, 6272000], "uso": ["urbano", "bosque"]}
    )
    serie = raster.extraer_series_puntos(indices, puntos)
    assert len(serie) == 2 * 3 * 2  # puntos x fechas x índices
    assert {"punto_id", "time", "indice", "valor", "uso"} <= set(serie.columns)

    tabla = raster.calcular_estadisticas(refl.isel(time=0), ["red", "nir08"])
    assert list(tabla["Banda"]) == ["red", "nir08"] and tabla["Media"].between(0, 1).all()


def test_lonlat_y_recorte():
    x, y = raster.lonlat_a_crs(-70.7, -33.65, "EPSG:32719")
    assert 3e5 < x < 4e5 and 6.2e6 < y < 6.3e6
    ds = _dataset_sintetico()
    lim = {"lon_min": -70.75, "lon_max": -70.65, "lat_min": -33.68, "lat_max": -33.62}
    recorte = raster.recortar_region(ds, lim)
    assert 0 < recorte.sizes["x"] <= ds.sizes["x"] and 0 < recorte.sizes["y"] <= ds.sizes["y"]


# ---------------------------------------------------------------- viz
def test_viz_matplotlib():
    refl = raster.convertir_a_reflectancia(_dataset_sintetico(), "landsat")
    imagen = refl.isel(time=0)
    rgb = viz.crear_imagen_rgb(imagen, ["red", "green", "blue"])
    assert rgb.shape == (20, 30, 3) and 0 <= rgb.min() and rgb.max() <= 1

    assert viz.mostrar_imagen(rgb, "RGB", mostrar=False) is not None
    ndvi = raster.calcular_ndvi(imagen, "landsat")
    assert viz.mostrar_imagen(ndvi, "NDVI", cmap="RdYlGn", barra="NDVI", mostrar=False) is not None
    fig = viz.mostrar_paneles([("NDVI", ndvi, {"cmap": "RdYlGn"}), ("RGB", rgb), ("otro", ndvi)], mostrar=False)
    assert len(fig.axes) >= 4
    assert viz.mostrar_mascaras({"a": ndvi > 0, "b": ndvi < 0}, mostrar=False) is not None
    assert viz.graficar_histogramas(imagen, ["red", "nir08", "swir16", "blue"], mostrar=False) is not None


def test_viz_capas():
    a = gpd.GeoDataFrame({"n": ["A"]}, geometry=[Polygon([(0, 0), (2, 0), (2, 2)])])
    b = gpd.GeoDataFrame({"n": ["B"]}, geometry=[Point(1, 1).buffer(0.5)])
    assert viz.graficar_capas([(a, "red"), (b, "blue", 0.3)], "prueba", mostrar=False) is not None


def test_viz_plotly():
    refl = raster.convertir_a_reflectancia(_dataset_sintetico(), "landsat")
    indices = raster.calcular_indices(refl, "landsat", ("NDVI", "EVI"))
    datos = refl.assign(NDVI=indices["NDVI"], EVI=indices["EVI"])

    fig = viz.histograma_interactivo(refl.isel(time=0), ["red", "nir08"], mostrar=False)
    assert len(fig.data) == 2

    puntos = pd.DataFrame({"punto_id": ["a", "b"], "x": [335000, 345000], "y": [6279000, 6272000]})
    serie = raster.extraer_series_puntos(indices, puntos)
    assert len(viz.graficar_series_temporales(serie, orientacion="filas", mostrar=False).data) == 4
    assert len(viz.graficar_series_temporales(serie, orientacion="columnas", promediar=True, mostrar=False).data) == 4
    assert len(viz.graficar_series_puntos(indices, [[335000, 6279000]], ("NDVI",), mostrar=False).data) == 1

    for modo in ("RGB", "Falso color", "NDVI", "EVI"):
        assert viz.mostrar_imagen_interactiva(datos, modo, 1, mostrar=False) is not None


def test_viz_mapas():
    mapa = viz.mapa_bbox(BBOX)
    assert "folium" in type(mapa).__module__


def test_selector_region():
    selector = viz.SelectorRegion(np.zeros((4, 4, 3)), BBOX)
    assert selector.limites is None
    cuadrado = {"geometry": {"type": "Polygon", "coordinates": [[[-70.7, -33.65], [-70.7, -33.62], [-70.65, -33.62], [-70.65, -33.65]]]}}
    selector._al_dibujar(None, "created", cuadrado)
    assert selector.limites == {"lon_min": -70.7, "lon_max": -70.65, "lat_min": -33.65, "lat_max": -33.62}
