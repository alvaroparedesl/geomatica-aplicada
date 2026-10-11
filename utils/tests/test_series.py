"""
Pruebas de `utils/series.py` y de lo que se agregó a `utils/raster.py` para las series
(sin internet).

* Puntos: `cargar_puntos_csv` (columnas, id repetidos, lat y lon invertidas) y los
  puntos del taller (`datos/puntos_concepcion.csv`).
* `extraer_series_puntos` con ventana (`radio`) y con lon/lat, y los índices NDWI_AGUA y NDCI.
* El flujo Sentinel-2 completo con GeoTIFF locales: `cargar_ventanas` (llamada nativa
  `odc.stac.load` por punto) y `construir_series`. Los DN se escriben a partir de
  reflectancias conocidas; una fecha está toda nublada, otra a medias, y el offset de
  cada fecha viene de la regla de `utils.stac` (baseline 03.00 en 2020 y 05.xx en 2025),
  así que si algún paso aplicara un offset fijo, los índices saldrían mal.
* Análisis: `rolling_media`, `tendencia_lineal`, `climatologia_mensual`, `anomalias_zscore`.
* Parquet: ida y vuelta con tipos y metadatos, y la serie preparada del taller.
"""

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin

from utils import raster, series, stac

from .conftest import leer_items

EPSG = "EPSG:32718"
N = 12
X0, Y0 = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True).transform(-73.10, -36.75)
X0, Y0 = round(X0, -1), round(Y0, -1)
CENTRO = (X0 + 60, Y0 - 60)  # el centro del GeoTIFF (el punto cae en el borde entre píxeles)
REFL = {"B03": 0.06, "B04": 0.05, "B05": 0.06, "B08": 0.02}  # agua: NDWI 0,5; NDCI 0,0909; NDVI -0,4286


@pytest.fixture
def punto_central():
    lon, lat = Transformer.from_crs(EPSG, "EPSG:4326", always_xy=True).transform(*CENTRO)
    return pd.DataFrame({"id": ["A"], "nombre": ["Agua de prueba"], "lat": [lat], "lon": [lon], "tipo": ["agua"]})


# ---------------------------------------------------------------------------
# Puntos
# ---------------------------------------------------------------------------
def test_cargar_puntos_csv(tmp_path):
    ruta = tmp_path / "p.csv"
    ruta.write_text("id,nombre,lat,lon,tipo,nota\n1,Bahía,-36.7,-73.0,agua,x\n2,Humedal,-36.8,-73.1,humedal,y\n")
    puntos = series.cargar_puntos_csv(ruta)
    assert list(puntos.columns) == ["id", "nombre", "lat", "lon", "tipo", "nota"]
    assert list(puntos["id"]) == ["1", "2"] and puntos["lat"].dtype == float  # id es texto aunque parezca número
    assert series.cargar_puntos_csv(str(ruta)).equals(puntos)


@pytest.mark.parametrize("contenido,mensaje", [
    ("id,nombre,lat,lon\n1,a,-36,-73\n", "faltan las columnas"),
    ("id,nombre,lat,lon,tipo\n1,a,-36,-73,x\n1,b,-37,-73,x\n", "repetidos"),
    ("id,nombre,lat,lon,tipo\n1,a,-73,-36.7,x\n2,b,-100,-36,x\n", "invertidas"),
])
def test_cargar_puntos_csv_errores(tmp_path, contenido, mensaje):
    (tmp_path / "p.csv").write_text(contenido)
    with pytest.raises(ValueError, match=mensaje):
        series.cargar_puntos_csv(tmp_path / "p.csv")


def test_puntos_del_taller_estan_en_la_zona_de_estudio():
    puntos = series.cargar_puntos_csv()
    assert len(puntos) >= 5 and puntos["id"].is_unique
    assert puntos["lat"].between(-37.3, -36.5).all() and puntos["lon"].between(-73.4, -72.8).all()
    assert "humedal" in set(puntos["tipo"]) and (puntos["tipo"] != "humedal").sum() >= 4  # agua y un punto de tierra
    gdf = series.puntos_a_geodataframe(puntos)
    assert gdf.crs.to_epsg() == 4326 and gdf.geometry.iloc[0].x == puntos["lon"].iloc[0]


def test_crs_utm_y_bbox_punto():
    assert series.crs_utm(-73.05, -36.8) == "EPSG:32718" and series.crs_utm(10.0, 45.0) == "EPSG:32632"
    oeste, sur, este, norte = series.bbox_punto(-73.0, -36.8, 150)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    (x1, x2), (y1, y2) = to_utm.transform([oeste, este], [sur, norte])
    assert abs((x2 - x1) - 300) < 10 and abs((y2 - y1) - 300) < 10  # ~300 m (la escala UTM añade un 2 %)
    puntos = pd.DataFrame({"lon": [-73.0, -72.9], "lat": [-36.8, -36.7]})
    caja = series.bbox_puntos(puntos, 150)
    assert caja[0] < -73.0 and caja[2] > -72.9 and caja[1] < -36.8 and caja[3] > -36.7


# ---------------------------------------------------------------------------
# raster: extraer_series_puntos con ventana y lon/lat; índices nuevos
# ---------------------------------------------------------------------------
def _cubo(valores_por_fecha, n=11):
    """Dataset (time, y, x) de 11 x 11 píxeles de 10 m en UTM 18S, con un valor por fecha."""
    x = X0 + 5 + 10 * np.arange(n)
    y = Y0 - 5 - 10 * np.arange(n)
    datos = np.stack([np.full((n, n), v, dtype="float32") for v in valores_por_fecha])
    ds = xr.Dataset({"V": (("time", "y", "x"), datos)},
                    coords={"time": pd.date_range("2025-01-01", periods=len(valores_por_fecha)), "y": y, "x": x})
    return ds.rio.write_crs(EPSG)


def test_extraer_series_puntos_ventana_mediana_ignora_nan():
    ds = _cubo([1.0, 2.0, 3.0])
    ds["V"][1, :, :5] = np.nan  # fecha 2: mitad de la ventana enmascarada
    ds["V"][2, 5, 5] = 100.0  # fecha 3: un valor extremo en el centro
    puntos = pd.DataFrame({"punto_id": ["p"], "x": [X0 + 55], "y": [Y0 - 55], "tipo": ["agua"]})
    s = raster.extraer_series_puntos(ds, puntos, radio=40).sort_values("time")
    assert list(s["valor"]) == [1.0, 2.0, 3.0]  # la mediana ignora los NaN y el extremo
    assert s["n_validos"].iloc[0] > s["n_validos"].iloc[1] > 0
    assert (s["x"] == X0 + 55).all() and (s["tipo"] == "agua").all()
    media = raster.extraer_series_puntos(ds, puntos, radio=40, estadistico="media").sort_values("time")
    assert media["valor"].iloc[2] > 3.0  # la media sí lo siente
    solo_el_pixel = raster.extraer_series_puntos(ds, puntos).sort_values("time")  # sin radio: píxel más cercano
    assert list(solo_el_pixel["valor"]) == [1.0, 2.0, 100.0] and "n_validos" not in solo_el_pixel


def test_extraer_series_puntos_con_lon_lat_y_radio_menor_que_un_pixel():
    ds = _cubo([5.0, 6.0])
    lon, lat = Transformer.from_crs(EPSG, "EPSG:4326", always_xy=True).transform(X0 + 55, Y0 - 55)
    puntos = pd.DataFrame({"punto_id": ["p"], "lon": [lon], "lat": [lat]})
    s = raster.extraer_series_puntos(ds, puntos)  # sin x, y: usa el CRS del Dataset
    assert sorted(s["valor"]) == [5.0, 6.0]
    s = raster.extraer_series_puntos(ds, puntos, radio=1)  # ningún centro de píxel a 1 m: el más cercano
    assert sorted(s["valor"]) == [5.0, 6.0] and set(s["n_validos"]) == {1}
    with pytest.raises(ValueError, match="estadistico"):
        raster.extraer_series_puntos(ds, puntos, radio=20, estadistico="moda")


def test_extraer_series_puntos_ventana_con_dask():
    ds = _cubo([1.0, 2.0]).chunk({"time": 1, "y": 4, "x": 4})
    puntos = pd.DataFrame({"punto_id": ["p"], "x": [X0 + 55], "y": [Y0 - 55]})
    assert sorted(raster.extraer_series_puntos(ds, puntos, radio=30)["valor"]) == [1.0, 2.0]


def test_indices_de_agua():
    refl = xr.Dataset({"B03": xr.DataArray([[0.06]]), "B04": xr.DataArray([[0.05]]),
                       "B05": xr.DataArray([[0.06]]), "B08": xr.DataArray([[0.02]])})
    assert raster.calcular_ndwi_agua(refl, "sentinel-2").item() == pytest.approx(0.5)
    assert raster.calcular_ndci(refl, "sentinel-2").item() == pytest.approx(1 / 11)
    indices = raster.calcular_indices(refl, "sentinel-2", ("NDWI_AGUA", "NDCI", "NDVI"))
    assert list(indices.data_vars) == ["NDWI_AGUA", "NDCI", "NDVI"]
    tierra = xr.Dataset({k: xr.DataArray([[v]]) for k, v in {"green": 0.1, "nir": 0.4}.items()})
    tierra = tierra.rename({"green": "B03", "nir": "B08"})
    assert raster.calcular_ndwi_agua(tierra, "sentinel-2").item() < 0  # tierra y vegetación: negativo
    landsat = xr.Dataset({"green": xr.DataArray([[0.06]]), "red": xr.DataArray([[0.05]])})
    with pytest.raises(ValueError, match="solo existe en Sentinel-2"):
        raster.calcular_ndci(landsat, "landsat")


# ---------------------------------------------------------------------------
# Flujo Sentinel-2 completo con GeoTIFF locales
# ---------------------------------------------------------------------------
def _escribir(ruta, valor, dtype, cubierto=None):
    perfil = {"driver": "GTiff", "width": N, "height": N, "count": 1, "dtype": dtype, "crs": EPSG,
              "transform": from_origin(X0, Y0, 10, 10), "nodata": 0}
    datos = np.full((1, N, N), valor, dtype=dtype)
    if cubierto is not None:
        datos[0, :, :cubierto[0]] = cubierto[1]
    with rasterio.open(ruta, "w", **perfil) as destino:
        destino.write(datos)


def _items_locales(carpeta):
    """Ítems reales (grabados) cuyos archivos son GeoTIFF locales con reflectancia conocida.

    Escena 1 (2020-12-04, baseline 03.00): despejada. Escena 2 (2021-12-14): toda nublada.
    Escena 3 (2025-02-16): nublada en la mitad izquierda. Escena 4 (2025-03-18): despejada.
    """
    items = stac.preparar_items(
        [i for a in ("pc_2020-12-04", "pc_2021-12-14", "pc_2025-02-16", "pc_2025-03-18") for i in leer_items(a)],
        verbose=False)
    scl = {0: (4, None), 1: (9, None), 2: (4, (N // 2, 9)), 3: (4, None)}
    assert len(items) == 4
    for k, item in enumerate(items):
        offset = stac.offset_sentinel2(item)
        for banda, r in REFL.items():
            ruta = carpeta / f"{item.id}_{banda}.tif"
            _escribir(ruta, round((r - offset) / 0.0001), "uint16")
            if banda not in item.assets:  # los ítems grabados solo traen B02, B04, B08 y SCL
                item.assets[banda] = pystac.Asset.from_dict(item.assets["B04"].to_dict())
            item.assets[banda].href = ruta.as_uri()
        valor, mitad = scl[k]
        ruta = carpeta / f"{item.id}_SCL.tif"
        _escribir(ruta, valor, "uint8", mitad)
        item.assets["SCL"].href = ruta.as_uri()
        item.properties["eo:cloud_cover"] = [3.0, 80.0, 40.0, 5.0][k]
    return items


@pytest.fixture
def ventanas(tmp_path, punto_central):
    items = _items_locales(tmp_path)
    return items, series.cargar_ventanas(items, punto_central, radio_m=50, verbose=False), punto_central


def test_cargar_ventanas_sigue_el_patron_nativo(ventanas):
    items, dic, puntos = ventanas
    ds = dic["A"]
    assert list(dic) == ["A"] and ds.sizes["time"] == 4
    assert {"B03", "B04", "B05", "B08", "SCL"} == set(ds.data_vars)  # renombradas, en DN
    assert ds["B04"].dtype == np.float32 and ds.sizes["x"] <= 12 and ds.sizes["x"] >= 9  # ventana de ~100 m
    assert ds["offset_reflectancia"].values.tolist() == [0.0, -0.1, -0.1, -0.1]  # baseline 03.00 en 2020
    assert ds["nubes_escena"].values.tolist() == [3.0, 80.0, 40.0, 5.0]


def test_cargar_ventanas_sin_imagenes_en_el_punto(ventanas):
    items, _, puntos = ventanas
    lejos = puntos.assign(lon=-70.0, lat=-33.0)
    with pytest.raises(ValueError, match="Ninguna imagen cubre"):
        series.cargar_ventanas(items, lejos, radio_m=50, verbose=False)


def test_agregar_nubes_escena_exige_los_mismos_items(ventanas):
    items, dic, _ = ventanas
    with pytest.raises(ValueError, match="no corresponden"):
        series.agregar_nubes_escena(dic["A"], items[:2])


def test_construir_series_aplica_mascara_offset_e_indices(ventanas):
    items, dic, puntos = ventanas
    tabla = series.construir_series(dic, puntos, radio_m=50)
    assert list(tabla.columns) == series.COLUMNAS_SERIES and len(tabla) == 4
    assert list(tabla["id"].unique()) == ["A"] and tabla["nombre"].iloc[0] == "Agua de prueba"
    assert tabla["fecha"].is_monotonic_increasing and str(tabla["fecha"].iloc[0])[:10] == "2020-12-04"
    despejada, nublada, mitad, despejada2 = (tabla.iloc[i] for i in range(4))
    for fila in (despejada, despejada2):  # el offset de cada fecha vuelve a dar las reflectancias de partida
        assert fila["ndwi"] == pytest.approx(0.5, abs=1e-3) and fila["ndci"] == pytest.approx(1 / 11, abs=1e-3)
        assert fila["b04"] == pytest.approx(0.05, abs=1e-4) and fila["ndvi"] == pytest.approx(-0.4286, abs=1e-3)
        assert fila["pixeles_validos"] == 1.0
    assert np.isnan(nublada[["ndwi", "ndci", "b04", "ndvi"]].to_numpy(dtype=float)).all()
    assert nublada["pixeles_validos"] == 0.0 and nublada["nubes_escena"] == 80.0
    assert mitad["pixeles_validos"] == pytest.approx(0.5, abs=0.12) and mitad["ndwi"] == pytest.approx(0.5, abs=1e-3)
    assert tabla["offset_reflectancia"].tolist() == [0.0, -0.1, -0.1, -0.1]
    assert tabla["n_pixeles"].min() > 20


def test_construir_series_con_fraccion_minima_dejan_nan_la_fecha_a_medias(ventanas):
    _, dic, puntos = ventanas
    tabla = series.construir_series(dic, puntos, radio_m=50, fraccion_minima=0.9)
    assert tabla["ndwi"].isna().tolist() == [False, True, True, False]
    assert len(tabla) == 4  # las fechas se conservan, con el índice en NaN
    resumen = series.resumen_validez(tabla)
    assert resumen.loc["A", "fechas"] == 4 and resumen.loc["A", "fechas_validas"] == 2
    assert resumen.loc["A", "porcentaje_valido"] == 50 and resumen.loc["A", "nubes_escena_media"] == pytest.approx(32.0)


def test_construir_series_descarta_fechas_sin_cobertura(ventanas):
    _, dic, puntos = ventanas
    ds = dic["A"].copy(deep=True)
    ds["SCL"][0] = 0  # la escena no cubre la ventana: SCL 0 (sin datos) en todos los píxeles
    tabla = series.construir_series({"A": ds}, puntos, radio_m=50)
    assert len(tabla) == 3 and str(tabla["fecha"].iloc[0])[:10] != "2020-12-04"


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------
def test_guardar_y_cargar_series(tmp_path, ventanas):
    _, dic, puntos = ventanas
    tabla = series.construir_series(dic, puntos, radio_m=50)
    ruta = tmp_path / "s.parquet"
    series.guardar_series(tabla, ruta, {"titulo": "Prueba", "licencia": "CC-BY-4.0", "columnas": {"ndwi": "agua"}})
    leida = series.cargar_series(ruta)
    pd.testing.assert_frame_equal(leida, tabla)
    assert leida["fecha"].dtype.kind == "M" and leida["id"].dtype == tabla["id"].dtype
    meta = series.leer_metadatos_series(ruta)
    assert meta["titulo"] == "Prueba" and meta["licencia"] == "CC-BY-4.0" and '"ndwi"' in meta["columnas"]
    assert "pandas" not in meta


def test_serie_preparada_del_taller():
    """La tabla `datos/series_sentinel2_concepcion.parquet` cumple el contrato que usan los cuadernos."""
    tabla = series.cargar_series()
    puntos = series.cargar_puntos_csv()
    assert list(tabla.columns) == series.COLUMNAS_SERIES
    assert set(tabla["id"]) == set(puntos["id"])
    assert tabla["fecha"].min() >= pd.Timestamp("2025-10-01") and tabla["fecha"].max() < pd.Timestamp("2026-10-01")
    assert tabla["offset_reflectancia"].isin([0.0, -0.1]).all()
    assert tabla["pixeles_validos"].between(0, 1).all() and tabla["nubes_escena"].between(0, 100).all()
    assert tabla[["ndwi", "ndci", "ndvi"]].stack().dropna().between(-1, 1).all()
    assert not tabla.duplicated(["id", "fecha"]).any()
    resumen = series.resumen_validez(tabla)
    assert (resumen["fechas_validas"] >= 60).all(), resumen  # ~80 de ~100 fechas por punto
    meta = series.leer_metadatos_series()
    assert {"titulo", "fuente", "licencia", "periodo"} <= set(meta)


# ---------------------------------------------------------------------------
# Análisis de una serie
# ---------------------------------------------------------------------------
def _serie_irregular(n=60, pendiente=2.0, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.DatetimeIndex(sorted(pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.choice(1000, n, replace=False), "D")))
    anios = (fechas - fechas[0]).days / 365.25
    return pd.Series(10 + pendiente * anios + rng.normal(0, 0.1, n), index=fechas, name="v")


def test_tendencia_lineal_recupera_la_pendiente():
    serie = _serie_irregular()
    t = series.tendencia_lineal(serie)
    assert t.pendiente_anual == pytest.approx(2.0, abs=0.05) and t.intercepto == pytest.approx(10, abs=0.1)
    assert t.r2 > 0.99 and t.p_valor < 1e-6 and t.n == 60
    assert t.ajuste.index.equals(serie.index) and t.ajuste.iloc[0] == pytest.approx(t.intercepto)
    con_nan = serie.copy()
    con_nan.iloc[[3, 7]] = np.nan
    assert series.tendencia_lineal(con_nan).n == 58
    with pytest.raises(ValueError, match="al menos 3"):
        series.tendencia_lineal(serie.iloc[:2])


def test_rolling_media_con_fechas_irregulares():
    serie = pd.Series([1.0, 3.0, 5.0, 100.0], index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05", "2025-06-01"]))
    r = series.rolling_media(serie, "5D", minimo=2)
    assert r.iloc[1] == 3.0 and r.iloc[0] == 2.0  # la ventana centrada de 5 días
    assert np.isnan(r.iloc[3])  # el valor aislado no llega al mínimo de observaciones
    assert series.rolling_media(serie, 2, minimo=2, centrada=False).iloc[1] == 2.0  # ventana por observaciones
    desordenada = series.rolling_media(serie.iloc[::-1], "5D", minimo=2)
    assert desordenada.index.is_monotonic_increasing


def test_climatologia_mensual():
    fechas = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    serie = pd.Series(fechas.month.to_numpy() + 0.0, index=fechas)
    c = series.climatologia_mensual(serie)
    assert list(c.index) == list(range(1, 13)) and list(c["media"]) == [float(m) for m in range(1, 13)]
    assert (c["desv"] == 0).all() and c.loc[1, "n"] == 4 * 31
    assert series.climatologia_mensual(serie, base=(2021, 2021)).loc[2, "n"] == 28
    parcial = series.climatologia_mensual(serie[serie.index.month <= 6])
    assert parcial.loc[7:, "media"].isna().all()  # los meses sin datos quedan NaN


def test_anomalias_zscore_por_mes_y_total():
    fechas = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(1)
    estacion = 5 * np.sin(2 * np.pi * fechas.dayofyear.to_numpy() / 365.25)
    serie = pd.Series(estacion + rng.normal(0, 1, len(fechas)), index=fechas, name="sst")
    serie.loc["2020-07-10"] += 6  # un día muy anómalo
    z = series.anomalias_zscore(serie)
    assert z.name == "sst_z" and z.index.equals(serie.index)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.1  # estandarizada
    assert z.loc["2020-07-10"] > 4.5
    z_total = series.anomalias_zscore(serie, referencia="total")
    assert z_total.loc["2020-07-10"] < 4.5 and z_total.std() == pytest.approx(1, abs=0.01)  # el ciclo estacional queda dentro de z
    base = series.anomalias_zscore(serie, base=(1991, 2000))
    assert base.index.equals(serie.index)
    with pytest.raises(ValueError, match="referencia"):
        series.anomalias_zscore(serie, referencia="dia")


def test_anomalias_zscore_con_un_solo_valor_por_mes_da_nan():
    serie = pd.Series(np.arange(12.0), index=pd.date_range("2025-01-15", periods=12, freq="MS") + pd.Timedelta(days=14))
    assert series.anomalias_zscore(serie).isna().all()  # un solo año: no hay desviación por mes
    assert series.anomalias_zscore(serie, referencia="total").notna().all()


# ---------------------------------------------------------------------------
# Visualización (folium y hvplot se importan solo dentro de las funciones)
# ---------------------------------------------------------------------------
def _puntos_viz():
    return pd.DataFrame({"id": ["a", "b", "c"], "nombre": ["Bahía A", "Laguna B", "Bahía C"],
                         "lat": [-36.69, -36.85, -36.75], "lon": [-73.0, -73.1, -73.17], "tipo": ["bahia", "laguna", "bahia"]})


def test_mapa_puntos():
    pytest.importorskip("folium")
    from utils.viz import mapa_puntos

    mapa = mapa_puntos(_puntos_viz(), radio_m=150)
    html = mapa.get_root().render()
    for nombre in ("Bahía A", "Laguna B", "Bahía C"):
        assert nombre in html
    assert "bahia (2)" in html and "laguna (1)" in html  # una capa por tipo
    assert html.count("L.circle(") >= 3 and "Esri" in html and "OpenStreetMap" in html
    una = mapa_puntos(_puntos_viz().iloc[:1], zoom=12)
    assert una.options["zoom"] == 12


def test_serie_interactiva():
    pytest.importorskip("hvplot")
    import holoviews as hv

    from utils.viz import serie_interactiva

    fechas = pd.date_range("2025-10-01", periods=6, freq="10D")
    tabla = pd.DataFrame({"id": ["a"] * 6 + ["b"] * 6, "fecha": list(fechas) * 2,
                          "ndwi": [0.5, 0.4, np.nan, 0.45, 0.5, 0.3] + [0.1] * 6})
    grafico = serie_interactiva(tabla, "ndwi", titulo="NDWI")
    assert isinstance(grafico, hv.Overlay)
    figura = hv.render(grafico)  # se puede dibujar con Bokeh
    assert len(figura.renderers) >= 4  # líneas y marcadores de dos puntos
    solo_a = hv.render(serie_interactiva(tabla, "ndwi", puntos=["a"]))
    assert 0 < len(solo_a.renderers) < len(figura.renderers)
