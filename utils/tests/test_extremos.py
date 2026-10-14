"""
Pruebas de `utils/extremos.py`: método de Hobday (series diarias) y anomalías y episodios
con series irregulares de Landsat 8 y 9.

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

Además: la climatología contra valores conocidos, la comparación con `xmhw` (se omite si no
está instalado), la malla y el Isolation Forest. Para Landsat: la climatología por ventana, los
episodios (racha de observaciones, brecha máxima, nubes que no cortan la racha), la malla de 500 m
y los datos reales preparados del taller (calibración, huecos y cobertura).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import datos_taller
from utils.extremos import (
    aislamiento_forest,
    aplicar_climatologia,
    anomalias_observaciones,
    anomalia_malla,
    aislamiento_forest_observaciones,
    cargar_landsat,
    controlar_calidad_agua,
    episodios_por_punto,
    climatologia_malla,
    climatologia_ventana,
    detectar_episodios,
    frecuencia_anomalias_malla,
    serie_landsat,
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
# Landsat: series irregulares (climatología por ventana, anomalías y episodios)
# ---------------------------------------------------------------------------
def serie_irregular(fechas, valores):
    return pd.Series(np.asarray(valores, dtype=float), index=pd.DatetimeIndex(fechas), name="st_c")


def _landsat_sintetica(anios=12, semilla=3, nubes=0.4):
    """Un paso cada 8 días desde 2013, ciclo anual de 3 °C (máximo a fines de enero) más ruido, con 40 % de fechas nubladas (NaN)."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2013-04-01", periods=int(anios * 365.25 / 8), freq="8D")
    valores = 15 + 3 * np.sin(2 * np.pi * (fechas.dayofyear.to_numpy() + 61) / 365.25) + rng.normal(0, 0.5, len(fechas))
    valores[rng.random(len(fechas)) < nubes] = np.nan
    return pd.Series(valores, index=fechas, name="st_c")


def test_climatologia_ventana_junta_observaciones_de_todos_los_anios():
    s = _landsat_sintetica()
    clima = climatologia_ventana(s, base=(2013, 2024), suavizado=None)
    assert list(clima.columns) == ["media", "p90", "p10", "n"] and list(clima.index) == list(range(1, 367))
    # 12 años x unas 4 pasadas por ventana de 31 días x 60 % sin nubes: decenas de observaciones, no cientos
    assert 15 < clima["n"].median() < 40
    assert (clima["p90"] > clima["media"]).all() and (clima["media"] > clima["p10"]).all()
    # el ciclo anual se recupera: máximo en verano austral (enero a marzo), mínimo en invierno
    assert clima["media"].idxmax() < 100 or clima["media"].idxmax() > 330
    assert 150 < clima["media"].idxmin() < 280
    # solo cuentan los años de la base
    assert climatologia_ventana(s, base=(2013, 2016), suavizado=None)["n"].sum() < clima["n"].sum() / 2


def test_climatologia_ventana_no_inventa_dias_sin_observaciones_suficientes():
    s = _landsat_sintetica(anios=2)
    clima = climatologia_ventana(s, base=(2013, 2015), minimo_obs=50)
    assert clima["media"].isna().all()
    clima = climatologia_ventana(_landsat_sintetica(), minimo_obs=8, suavizado=15)
    assert clima["media"].notna().all()
    with pytest.raises(ValueError, match="impares"):
        climatologia_ventana(s, ventana=30)
    with pytest.raises(ValueError, match="datos entre"):
        climatologia_ventana(s, base=(1990, 1995))


def test_anomalias_de_observaciones_alinean_por_dia_del_anio():
    s = serie_irregular(["2020-03-01", "2021-03-01", "2020-07-10"], [2.0, np.nan, -3.0])
    tabla = anomalias_observaciones(s, CLIMA_PLANA)
    assert list(tabla["anomalia"].dropna()) == [2.0, -3.0]
    assert list(tabla["excede_calor"]) == [True, False, False] and list(tabla["excede_frio"]) == [False, False, True]


def episodios(fechas, valores, tipo="calor", **kw):
    return detectar_episodios(serie_irregular(fechas, valores), CLIMA_PLANA, tipo, **kw)


def test_episodio_de_observaciones_seguidas_con_su_duracion_en_dias():
    ev = episodios(["2021-03-01", "2021-03-09", "2021-03-17", "2021-03-25"], [0, 3, 2, 0])
    assert len(ev) == 1
    fila = ev.iloc[0]
    assert fila["inicio"] == pd.Timestamp("2021-03-09") and fila["fin"] == pd.Timestamp("2021-03-17")
    assert fila["n_obs"] == 2 and fila["duracion"] == 9  # cota mínima: de la primera a la última observación
    assert fila["dia_pico"] == pd.Timestamp("2021-03-09") and fila["intensidad_max"] == 3 and fila["intensidad_media"] == 2.5
    assert not fila["en_curso"]


def test_una_sola_observacion_no_es_episodio_salvo_que_se_pida():
    fechas, valores = ["2021-03-01", "2021-03-09", "2021-03-17"], [0, 3, 0]
    assert episodios(fechas, valores).empty
    assert len(episodios(fechas, valores, min_obs=1)) == 1


def test_una_observacion_normal_en_medio_corta_el_episodio():
    assert episodios(["2021-03-01", "2021-03-09", "2021-03-17"], [2, 0, 2]).empty


def test_las_nubes_no_cortan_el_episodio_pero_un_hueco_largo_si():
    # una fecha nublada (NaN) entre dos observaciones anómalas: sigue siendo un episodio
    ev = episodios(["2021-03-01", "2021-03-09", "2021-03-17"], [2, np.nan, 2])
    assert len(ev) == 1 and ev.loc[0, "n_obs"] == 2 and ev.loc[0, "duracion"] == 17
    # 24 días entre observaciones: todavía se une; 25: ya no
    assert len(episodios(["2021-03-01", "2021-03-25"], [2, 2])) == 1
    assert episodios(["2021-03-01", "2021-03-26"], [2, 2]).empty
    assert len(episodios(["2021-03-01", "2021-03-26"], [2, 2], brecha_max_dias=30)) == 1


def test_episodios_de_frio_y_episodio_en_curso():
    ev = episodios(["2021-03-01", "2021-03-09", "2021-03-17"], [0, -2, -4], tipo="frio")
    assert len(ev) == 1 and ev.loc[0, "en_curso"] and ev.loc[0, "intensidad_max"] == -4
    assert evento_en_curso(ev)["inicio"] == pd.Timestamp("2021-03-09")
    assert episodios(["2021-03-01", "2021-03-09"], [-2, -4], tipo="calor").empty
    with pytest.raises(ValueError, match="tipo"):
        episodios(["2021-03-01"], [1], tipo="otro")


def test_dos_episodios_separados_por_una_observacion_normal():
    fechas = pd.date_range("2021-03-01", periods=7, freq="8D")
    ev = episodios(fechas, [2, 2, 0, 3, 3, 3, 0])
    assert list(ev["n_obs"]) == [2, 3] and list(ev["duracion"]) == [9, 17]
    assert resumen_eventos(ev).loc[0, "n_eventos"] == 2


def test_episodio_inyectado_en_serie_sintetica():
    s = _landsat_sintetica(nubes=0.2)
    clima = climatologia_ventana(s, base=(2013, 2018))
    # 4 observaciones seguidas muy cálidas en 2022
    posiciones = np.flatnonzero((s.index >= "2022-01-10") & s.notna().to_numpy())[:4]
    s.iloc[posiciones] += 4
    ev = detectar_episodios(s, clima, "calor")
    inyectado = ev[(ev["inicio"] >= "2022-01-01") & (ev["inicio"] <= "2022-03-01")]
    assert len(inyectado) == 1 and inyectado.iloc[0]["n_obs"] >= 4 and inyectado.iloc[0]["intensidad_max"] > 3


def test_isolation_forest_de_observaciones_marca_la_rara_y_respeta_las_nubes():
    pytest.importorskip("sklearn")
    s = _landsat_sintetica(nubes=0.2)
    tabla = pd.DataFrame({"st_c": s, "rojo": 0.02 + 0.001 * np.random.default_rng(0).normal(size=len(s))})
    tabla.iloc[100, 0] = 30.0  # una observación absurda (y su reflectancia normal)
    tabla.iloc[100, 1] = 0.02
    clima = {"st_c": climatologia_ventana(s)}
    res = aislamiento_forest_observaciones(tabla, ["st_c", "rojo"], clima, contaminacion=0.05)
    assert res.loc[tabla.index[100], "atipico"]
    assert not res.loc[tabla["st_c"].isna(), "atipico"].any() and res.loc[tabla["st_c"].isna(), "puntaje"].isna().all()
    assert res["atipico"].sum() == pytest.approx(0.05 * tabla["st_c"].notna().sum(), abs=3)
    repetido = aislamiento_forest_observaciones(tabla, ["st_c", "rojo"], clima, contaminacion=0.05)
    assert res["atipico"].equals(repetido["atipico"])


def _malla_landsat_sintetica():
    s = _landsat_sintetica(nubes=0.0)
    base = s.to_numpy()
    datos = np.stack([np.stack([base, base + 1.0], axis=-1),            # y = 0: dos píxeles de mar (el segundo, 1 °C más cálido)
                      np.full((len(base), 2), np.nan)], axis=1)         # y = 1: píxeles sin ningún dato
    return xr.DataArray(datos, dims=("time", "y", "x"), coords={"time": s.index, "y": [5.0e6, 5.0e6 - 500], "x": [6.0e5, 6.0e5 + 500]},
                        name="st_c")


def test_climatologia_de_malla_coincide_con_la_de_un_punto():
    malla = _malla_landsat_sintetica()
    clima = climatologia_malla(malla, base=(2013, 2024), paso=8)
    assert set(clima.data_vars) == {"media", "p90", "p10", "n"} and clima.sizes["centro"] == len(range(1, 367, 8))
    punto = climatologia_ventana(malla.isel(y=0, x=0).to_series(), base=(2013, 2024), suavizado=None)
    centros = clima["centro"].values
    esperada = punto.loc[centros]
    np.testing.assert_allclose(clima["media"].isel(y=0, x=0).values, esperada["media"].values, atol=1e-4)
    np.testing.assert_allclose(clima["p90"].isel(y=0, x=0).values, esperada["p90"].values, atol=1e-4)
    assert (clima["n"].isel(y=0, x=0).values == esperada["n"].values).all()
    np.testing.assert_allclose(clima["media"].isel(y=0, x=1).values, clima["media"].isel(y=0, x=0).values + 1.0, atol=1e-4)
    assert clima["media"].isel(y=1).isnull().all() and (clima["n"].isel(y=1) == 0).all()


def test_anomalia_y_frecuencia_de_anomalias_en_la_malla():
    malla = _malla_landsat_sintetica()
    clima = climatologia_malla(malla, base=(2013, 2020), paso=8)
    anomalia = anomalia_malla(malla, clima)
    assert anomalia.dims == malla.dims and abs(float(anomalia.isel(y=0, x=0).mean())) < 0.2
    frec = frecuencia_anomalias_malla(malla, clima, "calor")
    assert set(frec.data_vars) == {"n_obs", "n_anomalas", "fraccion"}
    # en el periodo base, cerca del 10 % de las observaciones supera el p90 de su píxel
    assert float(frec["fraccion"].isel(y=0, x=0)) == pytest.approx(0.10, abs=0.04)
    assert frec["fraccion"].isel(y=1).isnull().all()  # sin datos: NaN, no 0
    # en un periodo más reciente con un calentamiento de 2 °C casi todo excede el p90
    calentada = malla.where(malla.time.dt.year < 2022, malla + 2.0)
    reciente = frecuencia_anomalias_malla(calentada, clima, "calor", periodo=(2022, 2024))
    assert float(reciente["fraccion"].isel(y=0, x=0)) > 0.8
    assert float(reciente["n_obs"].isel(y=0, x=0)) < float(frec["n_obs"].isel(y=0, x=0))
    frio = frecuencia_anomalias_malla(calentada, clima, "frio", periodo=(2022, 2024))
    assert float(frio["fraccion"].isel(y=0, x=0)) == 0.0
    with pytest.raises(ValueError, match="tipo"):
        frecuencia_anomalias_malla(malla, clima, "otro")





def test_abrir_dato_descarga_si_no_hay_copia_local(tmp_path, monkeypatch):
    class Respuesta:
        content = b"fecha,valor\n2020-01-01,15\n"

        def raise_for_status(self):
            pass

    pedidos = []
    monkeypatch.setattr(datos_taller, "CARPETA_DATOS", tmp_path / "no_existe")
    monkeypatch.setattr(datos_taller, "CARPETA_CACHE", tmp_path / "cache")
    import requests

    monkeypatch.setattr(requests, "get", lambda url, timeout: pedidos.append(url) or Respuesta())
    ruta = datos_taller.abrir_dato("prueba.csv")
    assert ruta.read_text().startswith("fecha,valor") and pedidos[0].endswith("/datos/prueba.csv")
    assert "raw.githubusercontent.com/alvaroparedesl/geomatica-aplicada" in pedidos[0]
    datos_taller.abrir_dato("prueba.csv")  # segunda vez: usa la copia, sin descargar
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


def _tabla_puntos_sintetica():
    s = _landsat_sintetica(nubes=0.2)
    filas = []
    for punto, tipo, rojo, ndwi in (("a", "bahia", 0.01, 0.8), ("b", "humedal", 0.06, -0.6)):
        sub = pd.DataFrame({"id": punto, "nombre": punto.upper(), "tipo": tipo, "fecha": s.index, "st_dn": 40000.0, "st_c": s.to_numpy(),
                            "verde": 0.02, "rojo": rojo, "nir": 0.01, "swir1": 0.01, "ndwi": ndwi, "ndvi": -0.3})
        filas.append(sub)
    tabla = pd.concat(filas, ignore_index=True)
    con_dato = s.dropna().index
    tabla.loc[(tabla["id"] == "a") & (tabla["fecha"] == con_dato[50]), ["rojo"]] = 0.9      # nube sin marcar sobre agua
    tabla.loc[(tabla["id"] == "a") & (tabla["fecha"] == con_dato[60]), ["swir1"]] = 0.3     # SWIR1 brillante sobre agua
    tabla.loc[(tabla["id"] == "b") & (tabla["fecha"] == con_dato[50]), ["rojo"]] = 0.9      # el humedal (tierra) no se toca
    return tabla


def test_control_de_calidad_descarta_solo_el_agua_sospechosa():
    tabla = _tabla_puntos_sintetica()
    limpia = controlar_calidad_agua(tabla)
    nuevas = (limpia["st_c"].isna() & tabla["st_c"].notna())
    assert nuevas.sum() == 2 and set(limpia.loc[nuevas, "id"]) == {"a"}
    assert limpia.loc[nuevas, ["st_dn", "rojo", "swir1"]].isna().all().all()  # todas las variables, no solo la temperatura
    assert len(limpia) == len(tabla) and (limpia["fecha"] == tabla["fecha"]).all()  # la fecha se conserva
    assert (limpia.loc[limpia["id"] == "b", "st_c"].dropna() == tabla.loc[tabla["id"] == "b", "st_c"].dropna()).all()
    assert tabla["st_c"].notna().sum() == controlar_calidad_agua(tabla, rojo_max=10, swir1_max=10)["st_c"].notna().sum()


def test_episodios_por_punto_une_los_puntos_y_respeta_los_tipos():
    tabla = _tabla_puntos_sintetica()
    posiciones = np.flatnonzero((tabla["id"] == "a") & (tabla["fecha"] >= "2022-01-10") & tabla["st_c"].notna())[:4]
    tabla.loc[tabla.index[posiciones], "st_c"] += 4
    todo = episodios_por_punto(tabla)
    assert set(todo["punto"]) <= {"a", "b"} and "nombre" in todo and (todo["punto"] == "a").any()
    assert list(todo.columns[:3]) == ["punto", "nombre", "tipo"]
    solo_agua = episodios_por_punto(tabla, tipos=["bahia"])
    assert set(solo_agua["punto"]) == {"a"} and len(solo_agua) == (todo["punto"] == "a").sum()
    serie = serie_landsat(tabla, "a")
    esperado = detectar_episodios(serie, climatologia_ventana(serie), "calor")
    assert len(solo_agua[solo_agua["tipo"] == "calor"]) == len(esperado)
    with pytest.raises(ValueError, match="No hay un punto"):
        serie_landsat(tabla, "no_existe")
    assert episodios_por_punto(tabla, tipos=["ninguno"]).empty


# ---------------------------------------------------------------------------
# Datos reales de Landsat (archivos del repositorio)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tabla_landsat():
    return cargar_landsat()


def test_cargar_landsat_puntos(tabla_landsat):
    t = tabla_landsat
    assert t["id"].nunique() == 6 and t["fecha"].min() >= pd.Timestamp("2013-03-01") and t["fecha"].max() >= pd.Timestamp("2026-09-01")
    assert set(t["plataforma"]) == {"landsat-8", "landsat-9"} and t[t["plataforma"] == "landsat-9"]["fecha"].min() >= pd.Timestamp("2021-10-01")
    # calibración: st_c es st_dn * 0.00341802 + 149 - 273.15, y no hay temperaturas imposibles
    con_dato = t.dropna(subset=["st_c"])
    np.testing.assert_allclose(con_dato["st_dn"] * 0.00341802 + 149.0 - 273.15, con_dato["st_c"], atol=1e-3)
    assert con_dato["st_c"].between(-43, 67).all() and con_dato["rojo"].dropna().between(0, 1).all()
    # una fecha con menos de la mitad de la ventana válida no trae valores, pero se conserva
    poco = t[t["pixeles_validos"] < 0.5]
    assert len(poco) > 100 and poco["st_c"].isna().all() and poco["ndwi"].isna().all()
    # el agua es agua (NDWI alto, rojo bajo) y el humedal es vegetación
    mediana = t.groupby("id")[["ndwi", "ndvi", "rojo", "st_c"]].median()
    assert (mediana.loc[["bahia_concepcion", "golfo_arauco", "san_vicente"], "ndwi"] > 0.3).all()
    assert mediana.loc["humedal_rocuant", "ndvi"] > 0.4 and mediana.loc["humedal_rocuant", "ndwi"] < 0
    assert mediana.loc["humedal_rocuant", "st_c"] > mediana.loc["bahia_concepcion", "st_c"] + 8
    with pytest.raises(ValueError, match="tipo"):
        cargar_landsat("otro")


def test_serie_landsat_conserva_los_pasos_con_nubes(tabla_landsat):
    st = serie_landsat(tabla_landsat, "bahia_concepcion")
    assert st.name == "st_c" and st.index.is_monotonic_increasing and not st.index.duplicated().any()
    assert 350 < len(st) < 420 and 200 < st.notna().sum() < 260  # unos 6 de cada 10 pasos sin nubes
    intervalos = st.dropna().index.to_series().diff().dt.days.dropna()
    assert intervalos.min() >= 8 and intervalos.median() == 16 and intervalos.max() > 100  # revisita de 8 días y huecos largos
    por_anio = st.groupby(st.index.year).apply(lambda x: x.notna().sum())
    assert por_anio.loc[2022:2025].min() >= 18 and por_anio.loc[2014:2021].max() <= 18  # Landsat 9 desde 2022


def test_control_de_calidad_sobre_los_datos_reales(tabla_landsat):
    limpia = controlar_calidad_agua(tabla_landsat)
    descartadas = (tabla_landsat["st_c"].notna() & limpia["st_c"].isna()).groupby(tabla_landsat["id"]).sum()
    assert descartadas["humedal_rocuant"] == 0 and 40 <= descartadas.sum() <= 60
    antes = serie_landsat(tabla_landsat, "golfo_arauco")
    despues = serie_landsat(limpia, "golfo_arauco")
    assert antes["2025-05-06"] < 0 and np.isnan(despues["2025-05-06"])  # la nube que la máscara no vio (rojo de 0,95)
    # sin el control, la desviación de las anomalías es mayor
    a0 = anomalias_observaciones(antes, climatologia_ventana(antes))["anomalia"].std()
    a1 = anomalias_observaciones(despues, climatologia_ventana(despues))["anomalia"].std()
    assert a1 < a0 - 0.2


def test_climatologia_y_episodios_de_la_bahia(tabla_landsat):
    st = serie_landsat(controlar_calidad_agua(tabla_landsat), "bahia_concepcion")
    clima = climatologia_ventana(st)
    assert clima["n"].min() >= 5 and clima["media"].notna().sum() > 330  # unos pocos días del año pueden quedar sin climatología (menos de 8 observaciones)
    assert clima["media"].idxmax() < 100 or clima["media"].idxmax() > 330  # máximo en verano austral
    assert 150 < clima["media"].idxmin() < 260  # mínimo en invierno
    calor, frio = detectar_episodios(st, clima, "calor"), detectar_episodios(st, clima, "frio")
    assert (len(calor), len(frio)) == (0, 3)
    assert list(frio["inicio"]) == [pd.Timestamp("2014-03-05"), pd.Timestamp("2018-09-24"), pd.Timestamp("2026-05-09")] and list(frio["n_obs"]) == [2, 2, 3]
    assert (frio["duracion"] == [17, 17, 17]).all() and not frio["en_curso"].any()


def test_episodios_de_todos_los_puntos_de_agua_y_el_episodio_en_curso(tabla_landsat):
    ep = episodios_por_punto(controlar_calidad_agua(tabla_landsat), tipos=["bahia", "pluma", "golfo", "laguna"])
    assert len(ep) == 23 and set(ep["punto"]) == {"bahia_concepcion", "biobio_pluma", "golfo_arauco", "laguna_grande", "san_vicente"}
    assert ep["n_obs"].between(2, 4).all() and (ep["duracion"] >= 9).all()
    actual = evento_en_curso(ep)
    assert actual["punto"] == "golfo_arauco" and actual["tipo"] == "calor" and actual["inicio"] == pd.Timestamp("2026-09-06")
    assert ep["en_curso"].sum() == 1 and (ep.loc[ep["tipo"] == "calor", "intensidad_max"] > 0).all()
    assert (ep.loc[ep["tipo"] == "frio", "intensidad_max"] < 0).all()
    # ningún episodio cruza un hueco de más de 24 días entre observaciones consecutivas
    for fila in ep.itertuples():
        serie = serie_landsat(controlar_calidad_agua(tabla_landsat), fila.punto).dropna()
        dentro = serie[fila.inicio:fila.fin]
        saltos = dentro.index.to_series().diff().dt.days.dropna()
        assert len(dentro) == fila.n_obs and (saltos <= 24).all()


def test_malla_landsat_real():
    malla = cargar_landsat("malla")
    assert malla.dims == ("time", "y", "x") and malla.dtype == np.float32
    assert malla["time"].to_index().is_monotonic_increasing and malla.sizes["time"] > 250
    assert malla.sizes["x"] > 100 and malla.sizes["y"] > 100 and float(malla["x"][1] - malla["x"][0]) == 500.0
    assert float(malla.min()) > -45 and float(malla.max()) < 70  # °C, calibrados
    verano = malla.sel(time=malla["time"].dt.month.isin([12, 1, 2])).mean("time")
    # la tierra se calienta más que el mar en verano: el cuantil 90 de la malla supera el 10 por varios grados
    assert float(verano.quantile(0.9)) - float(verano.quantile(0.1)) > 4
    clima = climatologia_malla(malla)
    frec = frecuencia_anomalias_malla(malla, clima, "calor")
    assert 0.05 < float(frec["fraccion"].median()) < 0.15  # por construcción, cerca del 10 % en el periodo base y alrededores

