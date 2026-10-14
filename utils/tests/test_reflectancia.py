"""
Qué demuestra este archivo (sin internet): `utils.raster.convertir_a_reflectancia`
con cubos sintéticos (DN construidos con un offset conocido por fecha).

* El offset de la coordenada `offset_reflectancia` se aplica por fecha y se
  conserva al seleccionar una fecha o al enmascarar nubes.
* Un offset fijo (el -0.1 de antes) habría fallado en imágenes anteriores a 2022.
* DN 0 (sin datos) pasa a NaN, y los valores fuera de 0 a 1 también.
* Si se combinan fechas (por ejemplo `.median("time")`) antes de convertir, el offset
  se pierde: hay un ValueError claro que pide `agregar_offset` o `offset=...`.
* Landsat no cambia, y las máscaras de calidad SCL funcionan con "SCL" y "scl".
* Landsat 8/9: `convertir_a_temperatura` (DN a kelvin y a °C, DN 0 y valores imposibles a NaN) y la
  máscara de saturación `mascara_saturacion_landsat`.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import raster
from utils.raster import COORD_OFFSET

FECHAS = pd.to_datetime(["2021-12-14", "2022-01-23", "2022-01-31", "2025-02-16"])
OFFSETS = np.array([0.0, 0.0, -0.1, -0.1])
# Reflectancia "verdadera" de cada banda (agua oscura y vegetación)
REFL = {"B02": 0.012, "B04": 0.006, "B08": 0.35, "B11": 0.20}


def cubo_dn(offsets=OFFSETS, fechas=FECHAS, refl=REFL, ny=4, nx=5, scl=4):
    """DN como los entregaría el catálogo: DN = (reflectancia - offset) / 0.0001."""
    coords = {"time": fechas, "y": np.arange(ny)[::-1] * 10.0, "x": np.arange(nx) * 10.0}
    datos = {}
    for banda, r in refl.items():
        dn = np.round((r - offsets) / 0.0001).astype("uint16")
        datos[banda] = (("time", "y", "x"), np.broadcast_to(dn[:, None, None], (len(fechas), ny, nx)).copy())
    datos["SCL"] = (("time", "y", "x"), np.full((len(fechas), ny, nx), scl, dtype="uint8"))
    ds = xr.Dataset(datos, coords=coords)
    return ds.assign_coords({COORD_OFFSET: ("time", np.asarray(offsets, dtype=float))})


def test_offset_alineado_por_fecha_recupera_la_reflectancia():
    refl = raster.convertir_a_reflectancia(cubo_dn().drop_vars("SCL"), "sentinel-2")
    for banda, r in REFL.items():
        np.testing.assert_allclose(refl[banda].values, r, atol=1e-6)
    assert COORD_OFFSET not in refl.coords  # no se arrastra a la reflectancia


def test_offset_fijo_antiguo_hubiera_fallado():
    """Con el -0.1 fijo de antes, el agua de 2021 quedaba negativa (NaN) y el NDVI cambiaba."""
    ds = cubo_dn().drop_vars("SCL")
    mala = raster.convertir_a_reflectancia(ds, "sentinel-2", offset=-0.1)
    assert np.isnan(mala["B04"].isel(time=0)).all()  # 0.006 - 0.1 < 0
    buena = raster.convertir_a_reflectancia(ds, "sentinel-2")
    ndvi = raster.calcular_ndvi(buena, "sentinel-2")
    np.testing.assert_allclose(ndvi.values, (0.35 - 0.006) / (0.35 + 0.006), atol=1e-4)


def test_ndvi_identico_en_todas_las_fechas():
    ds = cubo_dn().drop_vars("SCL")
    ndvi = raster.calcular_ndvi(raster.convertir_a_reflectancia(ds, "sentinel-2"), "sentinel-2")
    medias = ndvi.mean(["y", "x"]).values
    assert np.ptp(medias) < 1e-4


def test_seleccion_de_una_fecha_conserva_su_offset():
    ds = cubo_dn().drop_vars("SCL")
    for i in range(len(FECHAS)):
        imagen = ds.isel(time=i)
        assert float(imagen[COORD_OFFSET]) == OFFSETS[i]
        refl = raster.convertir_a_reflectancia(imagen, "sentinel-2")
        assert float(refl["B04"].mean()) == pytest.approx(REFL["B04"], abs=1e-6)
    # sel con fecha en texto también
    refl = raster.convertir_a_reflectancia(ds.sel(time="2025-02-16"), "sentinel-2")
    assert float(refl["B08"].mean()) == pytest.approx(0.35, abs=1e-6)


def test_dataarray_con_offset():
    ds = cubo_dn()
    refl = raster.convertir_a_reflectancia(ds["B08"], "sentinel-2")
    np.testing.assert_allclose(refl.values, 0.35, atol=1e-6)


def test_offset_explicito_tiene_prioridad():
    ds = cubo_dn(offsets=np.zeros(4))
    refl = raster.convertir_a_reflectancia(ds.drop_vars("SCL"), "sentinel-2", offset=0.0)
    np.testing.assert_allclose(refl["B08"].values, 0.35, atol=1e-6)
    off = xr.DataArray(OFFSETS, dims="time", coords={"time": FECHAS})
    dn = cubo_dn().drop_vars(["SCL", COORD_OFFSET])
    refl = raster.convertir_a_reflectancia(dn, "sentinel-2", offset=off)
    np.testing.assert_allclose(refl["B02"].values, 0.012, atol=1e-6)


def test_sin_coordenada_usa_la_fecha_y_avisa():
    dn = cubo_dn().drop_vars(["SCL", COORD_OFFSET])
    with pytest.warns(UserWarning, match="offset según la fecha"):
        refl = raster.convertir_a_reflectancia(dn, "sentinel-2")
    # la regla por fecha acierta en estas fechas (ninguna es un reprocesamiento)
    np.testing.assert_allclose(refl["B08"].values, 0.35, atol=1e-6)


def test_sin_coordenada_ni_fecha_es_error():
    """Combinar fechas antes de convertir pierde el offset: mejor un error claro."""
    compuesto = cubo_dn().drop_vars("SCL").median("time")
    assert COORD_OFFSET not in compuesto.coords
    with pytest.raises(ValueError, match="antes de combinar fechas"):
        raster.convertir_a_reflectancia(compuesto, "sentinel-2")


def test_sin_datos_es_nan_aunque_el_offset_sea_cero():
    ds = cubo_dn().drop_vars("SCL")
    ds["B04"][0, 0, 0] = 0  # sin datos en una fecha con offset 0
    ds["B04"][3, 0, 0] = 0  # y en una con offset -0.1
    refl = raster.convertir_a_reflectancia(ds, "sentinel-2")
    assert np.isnan(refl["B04"][0, 0, 0]) and np.isnan(refl["B04"][3, 0, 0])
    assert int(refl["B04"].isnull().sum()) == 2


def test_rango_valido():
    ds = cubo_dn(refl={"B04": 0.05}).drop_vars("SCL")
    ds["B04"][:, 0, 0] = 1  # 0.0001 (offset 0) o -0.0999 (offset -0.1)
    ds["B04"][:, 0, 1] = 12000  # 1.2 o 1.1: saturado
    refl = raster.convertir_a_reflectancia(ds, "sentinel-2")["B04"]
    assert refl[0, 0, 0] == pytest.approx(0.0001)
    assert np.isnan(refl[3, 0, 0])
    assert refl[:, 0, 1].isnull().all()
    assert float(refl.min()) >= 0 and float(refl.max()) <= 1


def test_carga_perezosa_con_dask():
    ds = cubo_dn().drop_vars("SCL").chunk({"time": 1})
    refl = raster.convertir_a_reflectancia(ds, "sentinel-2")
    assert refl["B04"].chunks is not None
    np.testing.assert_allclose(refl["B04"].compute().values, 0.006, atol=1e-6)


def test_landsat_sin_cambios():
    ds = xr.Dataset({"red": (("time", "x"), np.array([[0, 10000, 43636]], dtype="uint16"))},
                    coords={"time": FECHAS[:1], "x": [0, 1, 2]})
    refl = raster.convertir_a_reflectancia(ds, "landsat")["red"].values[0]
    assert np.isnan(refl[0]) and refl[1] == pytest.approx(0.075) and refl[2] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- máscaras SCL
@pytest.mark.parametrize("nombre_scl", ["SCL", "scl"])
def test_preparar_reflectancia_con_scl(nombre_scl):
    ds = cubo_dn().rename({"SCL": nombre_scl})
    ds[nombre_scl][0, 0, 0] = 9  # nube alta
    ds[nombre_scl][2, 1, 1] = 3  # sombra
    ds[nombre_scl][3, 2, 2] = 0  # sin datos
    ds[nombre_scl][1, 3, 3] = 6  # agua: se conserva
    refl = raster.preparar_reflectancia(ds, "sentinel-2")
    assert nombre_scl not in refl
    assert int(refl["B04"].isnull().sum()) == 3
    assert not np.isnan(refl["B04"][1, 3, 3])
    np.testing.assert_allclose(refl["B04"].values[~np.isnan(refl["B04"].values)], 0.006, atol=1e-6)


@pytest.mark.parametrize("clase,valida", [(0, False), (1, False), (2, True), (3, False), (4, True), (5, True),
                                          (6, True), (7, True), (8, False), (9, False), (10, False), (11, True)])
def test_clases_scl(clase, valida):
    scl = xr.DataArray(np.array([clase], dtype="uint8"), dims="x")
    assert bool(raster.mascara_calidad_sentinel2(scl).values[0]) is valida


def test_enmascarar_conserva_offset_por_fecha():
    ds = cubo_dn()
    limpio = raster.enmascarar_nubes(ds, "sentinel-2")
    assert COORD_OFFSET in limpio.coords
    np.testing.assert_array_equal(limpio[COORD_OFFSET].values, OFFSETS)


# ---------------------------------------------------------------------------
# Landsat 8/9: temperatura de superficie y saturación
# ---------------------------------------------------------------------------
def test_temperatura_landsat_calibracion_en_kelvin_y_celsius():
    dn = xr.DataArray(np.array([[0, 33000, 40000, 65535]], dtype="uint16"))
    kelvin = raster.convertir_a_temperatura(dn, celsius=False)
    celsius = raster.convertir_a_temperatura(dn)
    # K = DN * 0.00341802 + 149
    assert float(kelvin[0, 1]) == pytest.approx(33000 * 0.00341802 + 149.0)
    assert float(kelvin[0, 2]) == pytest.approx(285.7208, abs=1e-3)
    assert float(celsius[0, 2]) == pytest.approx(285.7208 - 273.15, abs=1e-3)
    assert float(celsius[0, 1]) == pytest.approx(float(kelvin[0, 1]) - 273.15)


def test_temperatura_landsat_sin_dato_y_valores_imposibles_son_nan():
    dn = xr.DataArray(np.array([0, 20000, 40000, 65535], dtype="uint16"))  # 217 K y 373 K están fuera de 230 a 340 K
    celsius = raster.convertir_a_temperatura(dn)
    assert np.isnan(celsius.values[[0, 1, 3]]).all() and np.isfinite(celsius.values[2])


def test_temperatura_landsat_con_dataset_y_con_dn_float():
    ds = xr.Dataset({"lwir11": (("y", "x"), np.array([[40000.0, np.nan]]))})
    c = raster.convertir_a_temperatura(ds)
    assert float(c["lwir11"][0, 0]) == pytest.approx(12.57, abs=0.01) and np.isnan(float(c["lwir11"][0, 1]))


def test_mascara_de_saturacion_landsat():
    qa = xr.DataArray(np.array([0, 1 << 0, 1 << 6, 1 << 7, 1 << 9, 1 << 11], dtype="uint16"))
    # los bits 0 a 6 (bandas ópticas) y 11 (terreno) descartan; 7 (sin uso) y 9 (píxel perdido, sin uso en Landsat 8/9) no cuentan
    assert list(raster.mascara_saturacion_landsat(qa).values) == [True, False, False, True, True, False]
