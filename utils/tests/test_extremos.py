"""
Pruebas de `utils/extremos.py` (olas de calor y de frío marinas, método de Hobday).

Todo es offline. Los casos límite de la detección se prueban con series
sintéticas sobre una climatología plana (media 0, p90 = 1, p10 = -1), donde cada
día se puede razonar a mano:

* racha de exactamente 5 días (es evento) y de 4 (no lo es);
* brecha de 2 días (se unen) y de 3 (no se unen);
* primero se descartan las rachas cortas y después se unen (el orden importa);
* un vacío (NaN) corta la racha y una unión no lo cruza, ni aunque la brecha sea
  de 2 días; los días ausentes del índice cuentan como vacío;
* olas de frío (espejo de las de calor) y evento en curso;
* 29 de febrero (día 60 en el calendario de 366 días) en la climatología y en la duración.

Además: la climatología contra valores conocidos, los datos reales de OISST (los
conteos que se obtienen con 88 olas de calor y 111 de frío, el evento en curso
y el vacío de 2024-2026), la comparación con `xmhw` (se omite si no está
instalado), la malla y el Isolation Forest.
"""

import numpy as np
import pandas as pd
import pytest

from utils import datos_taller
from utils.extremos import (
    aislamiento_forest,
    aplicar_climatologia,
    cargar_oisst,
    climatologia_hobday,
    detectar_eventos,
    dia_del_anio_366,
    evento_en_curso,
    frecuencia_eventos,
    marcar_dias_evento,
    resumen_eventos,
    solapamiento,
)

CLIMA_PLANA = pd.DataFrame({"media": 0.0, "p90": 1.0, "p10": -1.0}, index=pd.RangeIndex(1, 367, name="dia"))


def serie_plana(n=60, inicio="2021-03-01", **tramos):
    """Serie diaria en 0 con tramos (posición inicial, valores): serie_plana(tramos=[(10, [2] * 5)])."""
    valores = np.zeros(n)
    for pos, vals in tramos.get("tramos", []):
        valores[pos:pos + len(vals)] = vals
    return pd.Series(valores, index=pd.date_range(inicio, periods=n, freq="D"))


def eventos(serie, tipo="calor", **kw):
    return detectar_eventos(serie, CLIMA_PLANA, tipo, **kw)


# ---------------------------------------------------------------------------
# Rachas, duración mínima y brechas
# ---------------------------------------------------------------------------
def test_racha_de_exactamente_5_dias_es_evento():
    s = serie_plana(tramos=[(10, [2] * 5)])
    ev = eventos(s)
    assert len(ev) == 1
    assert ev.loc[0, "inicio"] == s.index[10] and ev.loc[0, "fin"] == s.index[14] and ev.loc[0, "duracion"] == 5


def test_racha_de_4_dias_no_es_evento():
    assert eventos(serie_plana(tramos=[(10, [2] * 4)])).empty


def test_el_umbral_es_estricto():
    assert eventos(serie_plana(tramos=[(10, [1.0] * 8)])).empty  # igual al p90 no excede
    assert len(eventos(serie_plana(tramos=[(10, [1.01] * 8)]))) == 1


def test_brecha_de_2_dias_une_los_eventos():
    s = serie_plana(tramos=[(10, [2] * 5), (17, [2] * 5)])  # brecha: posiciones 15 y 16
    ev = eventos(s)
    assert len(ev) == 1
    assert ev.loc[0, "inicio"] == s.index[10] and ev.loc[0, "fin"] == s.index[21]
    assert ev.loc[0, "duracion"] == 12  # los días de la brecha cuentan


def test_brecha_de_3_dias_no_une():
    s = serie_plana(tramos=[(10, [2] * 5), (18, [2] * 5)])
    ev = eventos(s)
    assert list(ev["duracion"]) == [5, 5]


def test_brecha_maxima_configurable():
    s = serie_plana(tramos=[(10, [2] * 5), (18, [2] * 5)])
    assert list(eventos(s, brecha_max=3)["duracion"]) == [13]
    assert list(eventos(s, brecha_max=0)["duracion"]) == [5, 5]


def test_primero_se_descartan_las_rachas_cortas_y_despues_se_une():
    # 5 días, brecha 2, racha corta de 3 días, brecha 2, 5 días.
    # Si se uniera antes de filtrar, saldría un solo evento de 17 días.
    s = serie_plana(tramos=[(10, [2] * 5), (17, [2] * 3), (22, [2] * 5)])
    assert list(eventos(s)["duracion"]) == [5, 5]


def test_dos_rachas_cortas_no_se_unen_aunque_sumen_5():
    assert eventos(serie_plana(tramos=[(10, [2] * 3), (14, [2] * 3)])).empty


def test_duracion_minima_configurable():
    s = serie_plana(tramos=[(10, [2] * 3)])
    assert eventos(s).empty and len(eventos(s, duracion_min=3)) == 1


# ---------------------------------------------------------------------------
# Vacíos (NaN): nunca se puentean
# ---------------------------------------------------------------------------
def test_un_vacio_corta_la_racha():
    # 3 días excediendo, un vacío, 3 días excediendo: ninguna racha llega a 5
    s = serie_plana(tramos=[(10, [2, 2, 2, np.nan, 2, 2, 2])])
    assert eventos(s).empty


def test_un_vacio_corta_aunque_cada_lado_sea_evento():
    s = serie_plana(tramos=[(10, [2] * 5 + [np.nan] + [2] * 5)])
    ev = eventos(s)
    assert list(ev["duracion"]) == [5, 5]
    assert ev.loc[0, "fin"] == s.index[14] and ev.loc[1, "inicio"] == s.index[16]


def test_una_union_no_cruza_un_vacio_de_la_brecha():
    # brecha de 2 días (permitida) pero uno de los dos es un vacío: no se une
    s = serie_plana(tramos=[(10, [2] * 5 + [0, np.nan] + [2] * 5)])
    assert list(eventos(s)["duracion"]) == [5, 5]
    # y si la brecha es de 2 días sin vacío, sí se une
    s = serie_plana(tramos=[(10, [2] * 5 + [0, 0] + [2] * 5)])
    assert list(eventos(s)["duracion"]) == [12]


def test_un_vacio_largo_no_se_rellena():
    s = serie_plana(n=100, tramos=[(10, [2] * 6), (16, [np.nan] * 40), (56, [2] * 6)])
    ev = eventos(s)
    assert list(ev["duracion"]) == [6, 6]
    assert s.iloc[16:56].isna().all()


def test_los_dias_ausentes_del_indice_cuentan_como_vacio():
    s = serie_plana(tramos=[(10, [2] * 12)])
    sin_dias = s.drop(s.index[[13, 14]])  # faltan dos días en medio de la racha
    ev = eventos(sin_dias)
    assert list(ev["duracion"]) == [7]  # solo la parte de 7 días llega a 5; la de 3 no
    assert ev.loc[0, "inicio"] == s.index[15]


def test_vacio_al_final_no_deja_evento_en_curso():
    s = serie_plana(tramos=[(40, [2] * 8), (48, [np.nan] * 12)])
    ev = eventos(s)
    assert list(ev["duracion"]) == [8] and not ev.loc[0, "en_curso"]


def test_serie_toda_nan_o_sin_excedencias():
    assert eventos(serie_plana(tramos=[(0, [np.nan] * 60)])).empty
    assert eventos(serie_plana()).empty


# ---------------------------------------------------------------------------
# Frío, evento en curso, intensidad y categoría
# ---------------------------------------------------------------------------
def test_olas_de_frio_son_el_espejo_de_las_de_calor():
    calor = eventos(serie_plana(tramos=[(10, [2] * 5), (17, [2] * 5), (40, [2] * 4)]))
    frio = eventos(serie_plana(tramos=[(10, [-2] * 5), (17, [-2] * 5), (40, [-2] * 4)]), "frio")
    assert list(frio["duracion"]) == list(calor["duracion"]) == [12]
    assert frio.loc[0, "tipo"] == "frio" and frio.loc[0, "intensidad_max"] == -2.0


def test_el_calor_no_cuenta_como_frio_y_al_reves():
    assert eventos(serie_plana(tramos=[(10, [2] * 8)]), "frio").empty
    assert eventos(serie_plana(tramos=[(10, [-2] * 8)]), "calor").empty


def test_evento_en_curso_llega_al_ultimo_dia():
    s = serie_plana(tramos=[(50, [2] * 10)])
    ev = eventos(s)
    assert bool(ev.loc[0, "en_curso"]) and ev.loc[0, "fin"] == s.index[-1]
    assert evento_en_curso(ev)["duracion"] == 10
    assert evento_en_curso(eventos(serie_plana(tramos=[(10, [2] * 8)]))) is None


def test_intensidad_y_categoria():
    s = serie_plana(tramos=[(10, [1.5, 2.5, 3.5, 2.0, 1.2])])
    ev = eventos(s).loc[0]
    assert ev["intensidad_max"] == 3.5 and ev["dia_pico"] == s.index[12]
    assert ev["intensidad_media"] == pytest.approx(2.14) and ev["intensidad_acumulada"] == pytest.approx(10.7)
    assert ev["categoria"] == 3  # 3,5 veces la distancia media-umbral (1): categoría III
    muy_fuerte = eventos(serie_plana(tramos=[(10, [9.0] * 5)])).loc[0]
    assert muy_fuerte["categoria"] == 4  # la categoría máxima es IV
    assert eventos(serie_plana(tramos=[(10, [1.2] * 5)])).loc[0, "categoria"] == 1


def test_tipo_invalido():
    with pytest.raises(ValueError, match="calor"):
        eventos(serie_plana(), "tibio")


def test_marcar_dias_evento_y_solapamiento():
    s = serie_plana(tramos=[(10, [2] * 5), (17, [2] * 5)])
    marca = marcar_dias_evento(s, eventos(s))
    assert marca.sum() == 12 and marca.iloc[10] and marca.iloc[15] and not marca.iloc[9]
    otra = marca.copy()
    otra.iloc[10:12] = False
    resultado = solapamiento(marca, otra)
    assert resultado == {"dias_a": 12, "dias_b": 10, "comunes": 10, "jaccard": pytest.approx(10 / 12)}


# ---------------------------------------------------------------------------
# 29 de febrero y calendario de 366 días
# ---------------------------------------------------------------------------
def test_dia_del_anio_366():
    fechas = pd.DatetimeIndex(["2020-02-28", "2020-02-29", "2020-03-01", "2020-12-31",
                               "2021-02-28", "2021-03-01", "2021-12-31"])
    assert list(dia_del_anio_366(fechas)) == [59, 60, 61, 366, 59, 61, 366]


def test_evento_que_incluye_el_29_de_febrero():
    s = serie_plana(n=40, inicio="2020-02-15", tramos=[(10, [2] * 8)])  # 25 feb a 3 mar de 2020
    ev = eventos(s)
    assert ev.loc[0, "inicio"] == pd.Timestamp("2020-02-25") and ev.loc[0, "fin"] == pd.Timestamp("2020-03-03")
    assert ev.loc[0, "duracion"] == 8  # incluye el 29 de febrero
    assert pd.Timestamp("2020-02-29") in s.index
    sin = serie_plana(n=40, inicio="2021-02-15", tramos=[(10, [2] * 8)])
    assert eventos(sin).loc[0, "fin"] == pd.Timestamp("2021-03-04")


def test_el_dia_60_tiene_climatologia_aunque_la_base_no_tenga_29_de_febrero():
    indice = pd.date_range("2021-01-01", "2023-12-31", freq="D")  # tres años no bisiestos
    s = pd.Series(10 + np.sin(2 * np.pi * indice.dayofyear / 365.25), index=indice)
    clima = climatologia_hobday(s, base=(2021, 2023))
    assert len(clima) == 366 and clima.loc[60].notna().all()
    assert abs(clima.loc[60, "media"] - (clima.loc[59, "media"] + clima.loc[61, "media"]) / 2) < 0.01


def test_29_de_febrero_en_la_serie_se_compara_con_el_umbral_del_dia_60():
    indice = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    s = pd.Series(0.0, index=indice)
    clima = pd.DataFrame({"media": 0.0, "p90": 1.0, "p10": -1.0}, index=pd.RangeIndex(1, 367, name="dia"))
    clima.loc[60, "p90"] = 5.0  # solo el 29 de febrero tiene el umbral alto
    s["2020-02-29"] = 3.0
    s["2021-03-01"] = 3.0  # mismo valor un día después del 28 de febrero de un año no bisiesto
    tabla = aplicar_climatologia(s, clima)
    assert tabla.loc["2020-02-29", "p90"] == 5.0 and tabla.loc["2021-03-01", "p90"] == 1.0
    assert tabla.loc["2020-03-01", "p90"] == 1.0 and tabla.loc["2021-02-28", "p90"] == 1.0


# ---------------------------------------------------------------------------
# Climatología
# ---------------------------------------------------------------------------
def test_climatologia_de_una_serie_constante():
    indice = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    clima = climatologia_hobday(pd.Series(15.0, index=indice))
    assert list(clima.columns) == ["media", "p90", "p10"]
    assert np.allclose(clima.to_numpy(), 15.0)
    assert clima.index[0] == 1 and clima.index[-1] == 366


def test_climatologia_solo_usa_el_periodo_base():
    indice = pd.date_range("1985-01-01", "2024-12-31", freq="D")
    valores = np.where((indice.year >= 1991) & (indice.year <= 2020), 10.0, 99.0)
    clima = climatologia_hobday(pd.Series(valores, index=indice))
    assert np.allclose(clima.to_numpy(), 10.0)
    clima_otra = climatologia_hobday(pd.Series(valores, index=indice), base=(2021, 2024))
    assert np.allclose(clima_otra.to_numpy(), 99.0)


def test_climatologia_estacional_y_percentiles_ordenados():
    rng = np.random.default_rng(0)
    indice = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    estacion = 3 * np.sin(2 * np.pi * (indice.dayofyear.to_numpy() - 80) / 365.25)
    s = pd.Series(15 + estacion + rng.normal(0, 0.5, len(indice)), index=indice)
    clima = climatologia_hobday(s)
    assert (clima["p10"] < clima["media"]).all() and (clima["media"] < clima["p90"]).all()
    assert abs(clima.loc[171, "media"] - 18.0) < 0.1  # 3 sen(...) vale 3 hacia el día 171
    assert abs((clima["p90"] - clima["media"]).mean() - 1.28 * 0.5) < 0.1  # el p90 de una normal está a 1,28 sigma


def test_climatologia_ignora_los_vacios_y_falla_sin_base():
    indice = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    s = pd.Series(15.0, index=indice)
    s.iloc[1000:3000] = np.nan
    assert np.allclose(climatologia_hobday(s).to_numpy(), 15.0)
    with pytest.raises(ValueError, match="no tiene datos"):
        climatologia_hobday(pd.Series(1.0, index=pd.date_range("2000-01-01", "2001-12-31")), base=(2010, 2020))
    with pytest.raises(ValueError, match="impares"):
        climatologia_hobday(s, ventana=10)


def test_el_suavizado_cierra_el_año():
    indice = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    s = pd.Series(np.where(indice.dayofyear > 360, 20.0, 10.0), index=indice)  # escalón al final del año
    clima = climatologia_hobday(s)
    assert clima.loc[1, "media"] > 10.5  # el escalón se suaviza hacia el inicio del año


# ---------------------------------------------------------------------------
# Datos reales de OISST (archivos del repositorio)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sst():
    return cargar_oisst()


def test_cargar_oisst_punto(sst):
    assert sst.index.freqstr == "D" and sst.name == "sst_c"
    assert sst.index[0] == pd.Timestamp("1981-09-01") and sst.index[-1] >= pd.Timestamp("2026-09-01")
    assert sst.isna().sum() == 626  # 2024-06-20 a 2026-03-05 (624 días) y 2026-03-07 y -08
    assert sst["2024-06-20":"2026-03-05"].isna().all() and sst["2024-06-19"] > 0 and sst["2026-03-06"] > 0
    assert sst["2026-03-07":"2026-03-08"].isna().all()


def test_cargar_oisst_malla():
    malla = cargar_oisst("malla")
    assert malla.dims == ("time", "lat", "lon") and malla.sizes["lat"] == 24 and malla.sizes["lon"] == 20
    assert malla.time.to_index().freqstr == "D"  # calendario completo: los días ausentes son NaN
    assert malla.sel(time="2025-01-01").isnull().all()  # dentro del vacío
    assert malla.sel(time="2023-01-01").notnull().sum() > 100  # el mar tiene datos
    with pytest.raises(ValueError, match="tipo"):
        cargar_oisst("otro")


def test_hobday_sobre_oisst_concepcion(sst):
    """Resultado de referencia del taller (validado con xmhw): ver el docstring del módulo."""
    clima = climatologia_hobday(sst)
    calor, frio = detectar_eventos(sst, clima, "calor"), detectar_eventos(sst, clima, "frio")
    assert (len(calor), len(frio)) == (88, 111)
    mayores = calor.nlargest(3, "duracion")
    assert list(mayores["inicio"].dt.year) == [1997, 1998, 2017]
    assert list(mayores["duracion"]) == [63, 52, 44]
    # el evento en curso empieza el 2026-08-17, después del vacío, y llega al último día
    actual = evento_en_curso(calor)
    assert actual["inicio"] == pd.Timestamp("2026-08-17") and actual["fin"] == sst.index[-1]
    assert not frio["en_curso"].any()
    # ningún evento cruza el vacío de 2024-2026
    vacio = pd.Timestamp("2024-06-20"), pd.Timestamp("2026-03-05")
    for ev in (calor, frio):
        assert not ((ev["inicio"] <= vacio[1]) & (ev["fin"] >= vacio[0])).any()
    por_decada = resumen_eventos(calor, "decada")
    assert por_decada["n_eventos"].sum() == 88 and list(por_decada.index) == [1980, 1990, 2000, 2010, 2020]
    assert resumen_eventos(calor).loc[0, "n_eventos"] == 88


def test_resumen_de_eventos_vacio_y_por_anio():
    vacio = eventos(serie_plana())
    assert resumen_eventos(vacio).loc[0, "n_eventos"] == 0 and resumen_eventos(vacio, "anio").empty
    s = serie_plana(n=400, tramos=[(10, [2] * 6), (300, [2] * 7)])
    por_anio = resumen_eventos(eventos(s), "anio")
    assert list(por_anio.index) == [2021] and por_anio.loc[2021, "dias_totales"] == 13
    with pytest.raises(ValueError, match="por"):
        resumen_eventos(eventos(s), "mes")


def test_abrir_dato_descarga_si_no_hay_copia_local(tmp_path, monkeypatch):
    class Respuesta:
        content = b"fecha,sst_c\n2020-01-01,15\n"

        def raise_for_status(self):
            pass

    pedidos = []
    monkeypatch.setattr(datos_taller, "CARPETA_DATOS", tmp_path / "no_existe")
    monkeypatch.setattr(datos_taller, "CARPETA_CACHE", tmp_path / "cache")
    import requests

    monkeypatch.setattr(requests, "get", lambda url, timeout: pedidos.append(url) or Respuesta())
    ruta = datos_taller.abrir_dato("oisst_concepcion.csv")
    assert ruta.read_text().startswith("fecha,sst_c") and pedidos[0].endswith("/datos/oisst_concepcion.csv")
    assert "raw.githubusercontent.com/alvaroparedesl/geomatica-aplicada" in pedidos[0]
    datos_taller.abrir_dato("oisst_concepcion.csv")  # segunda vez: usa la copia, sin descargar
    assert len(pedidos) == 1


# ---------------------------------------------------------------------------
# Comparación con xmhw (se omite si no está instalado)
# ---------------------------------------------------------------------------
def _serie_sintetica(anios=30, semilla=1):
    rng = np.random.default_rng(semilla)
    indice = pd.date_range("1991-01-01", periods=int(365.25 * anios), freq="D")
    ruido = np.zeros(len(indice))
    impulsos = rng.normal(0, 0.3, len(indice))
    for i in range(1, len(indice)):
        ruido[i] = 0.9 * ruido[i - 1] + impulsos[i]
    valores = 15 + 3 * np.sin(2 * np.pi * (indice.dayofyear.to_numpy() - 80) / 365.25) + ruido
    valores[5000:5020] += 2.5  # ola de calor inyectada
    valores[7000:7015] -= 2.5  # ola de frío inyectada
    return pd.Series(valores, index=indice)


def test_hobday_coincide_con_xmhw_en_serie_sintetica():
    xr = pytest.importorskip("xarray")
    xmhw = pytest.importorskip("xmhw.xmhw")
    s = _serie_sintetica()
    s_nan = s.copy()
    s_nan.iloc[3000:3040] = np.nan  # con un vacío
    for serie in (s, s_nan):
        da = xr.DataArray(serie.values, dims="time", coords={"time": serie.index.values})
        clima = climatologia_hobday(serie)
        for tipo, columna, signo in (("calor", "p90", 1), ("frio", "p10", -1)):
            th = xmhw.threshold(da, climatologyPeriod=[1991, 2020], pctile=90, coldSpells=tipo == "frio")
            ev = xmhw.detect(da, th["thresh"], th["seas"], minDuration=5, joinGaps=True, maxGap=2, coldSpells=tipo == "frio")
            assert np.abs(signo * th["thresh"].squeeze().to_series().to_numpy() - clima[columna].to_numpy()).max() < 0.01
            assert np.abs(signo * th["seas"].squeeze().to_series().to_numpy() - clima["media"].to_numpy()).max() < 0.01
            mios = detectar_eventos(serie, clima, tipo)
            inicios = pd.to_datetime(ev.time_start.values.ravel())
            inicios = set(inicios[~pd.isna(inicios)])
            # Las curvas difieren hasta 0,007 °C: un día que queda a menos de eso del umbral puede
            # contarse distinto, así que se tolera algún evento de diferencia (y que xmhw cuente
            # como fin el primer día de un vacío, cosa que aquí no se hace).
            assert len(inicios ^ set(mios["inicio"])) <= 2
            assert abs(len(inicios) - len(mios)) <= 1
            assert serie[marcar_dias_evento(serie, mios)].notna().all()  # ningún evento incluye un vacío
    assert len(detectar_eventos(s, climatologia_hobday(s), "calor")) >= 1  # el evento inyectado aparece


# ---------------------------------------------------------------------------
# Malla
# ---------------------------------------------------------------------------
def _malla_sintetica():
    import xarray as xr

    s = _serie_sintetica(anios=8)
    base = s.to_numpy()
    datos = np.stack([np.stack([base, base + 0.1], axis=-1), np.full((len(base), 2), np.nan)], axis=1)  # (time, lat, lon)
    return xr.DataArray(datos, dims=("time", "lat", "lon"),
                        coords={"time": s.index, "lat": [-36.0, -36.25], "lon": [-73.5, -73.25]}, name="sst")


def test_frecuencia_de_eventos_en_una_malla():
    malla = _malla_sintetica()
    f = frecuencia_eventos(malla, "calor", base=(1991, 1998))
    serie = malla.isel(lat=0, lon=0).to_series()
    esperado = detectar_eventos(serie, climatologia_hobday(serie, base=(1991, 1998)), "calor")
    assert f["n_eventos"].dims == ("lat", "lon") and f["n_eventos"].isel(lat=0, lon=0) == len(esperado)
    assert f["dias_en_evento"].isel(lat=0, lon=0) == esperado["duracion"].sum()
    assert f["n_eventos"].isel(lat=1).isnull().all()  # fila de "tierra" (todo NaN)
    assert f["dias_con_dato"].isel(lat=0, lon=0) == len(serie)
    assert f["eventos_por_decada"].isel(lat=0, lon=0) == pytest.approx(len(esperado) / (len(serie) / 3652.5))
    g = frecuencia_eventos(malla, "calor", base=(1991, 1998), periodo=(1995, 1998))
    assert g["dias_con_dato"].isel(lat=0, lon=0) < f["dias_con_dato"].isel(lat=0, lon=0)
    assert g["n_eventos"].isel(lat=0, lon=0) <= f["n_eventos"].isel(lat=0, lon=0)


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
def test_isolation_forest_marca_el_evento_inyectado_y_respeta_los_vacios():
    pytest.importorskip("sklearn")
    s = _serie_sintetica()
    s.iloc[3000:3040] = np.nan
    resultado = aislamiento_forest(s, contaminacion=0.05)
    assert resultado["atipico"].mean() == pytest.approx(0.05, abs=0.01)
    inyectado = resultado.iloc[5002:5018]
    assert inyectado["atipico_calor"].mean() > 0.8 and not inyectado["atipico_frio"].any()
    assert resultado.iloc[7002:7013]["atipico_frio"].mean() > 0.8
    assert not resultado.iloc[3000:3040]["atipico"].any() and resultado.iloc[3000:3040]["puntaje"].isna().all()
    repetido = aislamiento_forest(s, contaminacion=0.05)
    assert resultado["atipico"].equals(repetido["atipico"])  # misma semilla, mismo resultado
    hobday = marcar_dias_evento(s, detectar_eventos(s, climatologia_hobday(s), "calor"))
    assert solapamiento(hobday, resultado["atipico_calor"])["comunes"] >= 15
