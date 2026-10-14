"""
Anomalías y eventos extremos de temperatura con series satelitales.

Este módulo tiene dos partes que responden a dos tipos de serie.

**A. Series irregulares de Landsat 8 y 9 (la que usa el taller).** Landsat pasa cada 8
días sobre un punto (con los dos satélites) y las nubes borran muchas fechas, así que la
serie tiene solo unas decenas de observaciones al año, a intervalos desiguales. No se
puede aplicar tal cual el método de Hobday (necesita una serie diaria sin huecos y 30
años de base), pero sí su idea: comparar cada observación con lo normal **para esa época
del año**.

1. `cargar_landsat`: la tabla de series (temperatura de superficie calibrada, reflectancia
   e índices por punto y fecha) o la malla de 500 m.
2. `climatologia_ventana`: para cada día del año, la media y los percentiles de las
   observaciones del periodo base que caen a +-15 días (se juntan todos los años).
3. `anomalias_observaciones` y `detectar_episodios`: una observación es anómala si supera
   el p90 (calor) o queda bajo el p10 (frío); un episodio son varias observaciones
   válidas seguidas con la misma señal, sin cruzar huecos largos (`brecha_max_dias`).
   No es un evento de Hobday: la duración es solo una cota mínima.
4. `aislamiento_forest_observaciones`: Isolation Forest sobre varias variables a la vez
   (temperatura, reflectancia roja, NDWI...).
5. `climatologia_malla` y `frecuencia_anomalias_malla`: lo mismo píxel por píxel.

**B. Series diarias sin huecos (método de Hobday et al. 2016).** Se conserva como método
de referencia, con pandas y numpy legibles, para series diarias largas (por ejemplo un
producto diario de temperatura del mar):

1. `climatologia_hobday`: media y percentiles 90 y 10 por día del año de un periodo base
   (1991-2020), juntando los días a +-5 días y suavizando con una media móvil de 31 días.
2. `detectar_eventos`: un día "excede" si supera el p90 (calor) o queda bajo el p10 (frío).
   Un evento es una racha de excedencias de al menos 5 días; dos eventos separados por 2
   días o menos se unen (primero se descartan las rachas cortas y después se unen).
3. `resumen_eventos`, `evento_en_curso`, `frecuencia_eventos` (una malla), `aislamiento_forest`.

Un día sin dato (NaN) **nunca se rellena ni se puentea**: no excede el umbral, así que corta
cualquier racha, y una unión de eventos no puede cruzar un vacío. Las pruebas
(`utils/tests/test_extremos.py`) comparan esta implementación con `xmhw`
(https://github.com/coecms/xmhw, implementación de la comunidad Hobday; `pip install
git+https://github.com/coecms/xmhw`, no está en PyPI) sobre una serie sintética, si está instalado.
Con `xmhw` el frío se pide con `coldSpells=True` y `pctile=90` (él invierte la serie).

Uso con Landsat:

    from utils.extremos import (cargar_landsat, serie_landsat, climatologia_ventana,
                                anomalias_observaciones, detectar_episodios)
    tabla = cargar_landsat()                              # una fila por punto y fecha
    st = serie_landsat(tabla, "bahia_concepcion")         # temperatura de superficie, °C
    clima = climatologia_ventana(st)                      # media, p10, p90 por día del año
    calor = detectar_episodios(st, clima, "calor")
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .datos_taller import abrir_dato

NOMBRES_CATEGORIA = {1: "I moderada", 2: "II fuerte", 3: "III severa", 4: "IV extrema"}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_landsat(tipo="puntos"):
    """
    Carga los datos de Landsat 8 y 9 preparados para el taller.

    Lee de `talleres/fair-series-temporales/datos/` si existe y, si no, de GitHub
    (ver `utils.datos_taller`). Los datos ya están calibrados: temperatura de
    superficie en °C y reflectancia de 0 a 1 (ver `utils.raster.convertir_a_temperatura`
    y `utils.raster.convertir_a_reflectancia`). Las fechas con nubes **no** se
    rellenan: tienen `pixeles_validos` bajo y las variables en NaN.

    Args:
        tipo (str): "puntos" (tabla de series de los seis puntos, desde
            `series_landsat_concepcion.parquet`) o "malla" (temperatura de
            superficie cada 500 m alrededor de la bahía de Concepción y el golfo de
            Arauco, desde `landsat_malla_concepcion.nc`).

    Returns:
        pandas.DataFrame o xarray.DataArray: "puntos" da una tabla con una fila por
        punto y fecha (columnas `id`, `nombre`, `tipo`, `lat`, `lon`, `fecha`,
        `plataforma`, `st_dn`, `st_c`, `verde`, `rojo`, `nir`, `swir1`, `ndwi`,
        `ndvi`, `nubes_escena`, `pixeles_validos`, `n_pixeles`). "malla" da el
        DataArray `st_c` (°C) con dimensiones (time, y, x) en UTM 18 S (EPSG:32718),
        con NaN donde hay nubes y solo las fechas con al menos 10 % de píxeles válidos.
    """
    if tipo == "puntos":
        tabla = pd.read_parquet(abrir_dato("series_landsat_concepcion.parquet"))
        tabla["fecha"] = pd.to_datetime(tabla["fecha"])
        return tabla
    if tipo == "malla":
        with xr.open_dataset(abrir_dato("landsat_malla_concepcion.nc")) as ds:
            return ds["st_c"].load().astype("float32")
    raise ValueError(f"tipo debe ser 'puntos' o 'malla', no {tipo!r}.")


def serie_landsat(tabla, punto, variable="st_c"):
    """
    Una variable de un punto como serie de pandas con índice de fechas (a medianoche UTC).

    Las fechas con nubes (menos de la mitad de la ventana válida) quedan como NaN: la
    serie conserva **todos los pasos del satélite**, así se ven los huecos.

    Args:
        tabla (pandas.DataFrame): Resultado de `cargar_landsat("puntos")`.
        punto (str): El `id` del punto, por ejemplo "bahia_concepcion".
        variable (str): Columna de la tabla: "st_c", "rojo", "ndwi", ...

    Returns:
        pandas.Series: Llamada como `variable`, ordenada por fecha.
    """
    if punto not in set(tabla["id"]):
        raise ValueError(f"No hay un punto con id {punto!r}: usa uno de {sorted(set(tabla['id']))}.")
    sub = tabla[tabla["id"] == punto]
    serie = pd.Series(sub[variable].to_numpy(dtype=float), index=sub["fecha"].dt.normalize(), name=variable)
    return serie.groupby(level=0).first().sort_index()  # si un día viene repetido, la primera observación que no es NaN


# ---------------------------------------------------------------------------
# Climatología de Hobday
# ---------------------------------------------------------------------------
def dia_del_anio_366(indice):
    """
    Día del año en un calendario de 366 días (el 29 de febrero siempre es el 60).

    En años no bisiestos, desde el 1 de marzo se suma 1: así el 1 de marzo es
    siempre el día 61 y cada fecha del calendario cae en el mismo número todos los
    años. Es la convención de Hobday et al. y de `xmhw`.

    Args:
        indice (pandas.DatetimeIndex): Fechas.

    Returns:
        numpy.ndarray: Enteros de 1 a 366.
    """
    dia = indice.dayofyear.to_numpy()
    return dia + (~indice.is_leap_year & (indice.month >= 3)).astype(int)


def _media_movil_circular(valores, ancho):
    """Media móvil centrada que cierra el año sobre sí mismo (el día 366 vecino del 1)."""
    mitad = ancho // 2
    ampliado = np.concatenate([valores[-mitad:], valores, valores[:mitad]])
    return np.convolve(ampliado, np.ones(ancho) / ancho, mode="valid")


def climatologia_hobday(serie, base=(1991, 2020), ventana=11, percentiles=(90, 10), suavizado=31):
    """
    Climatología diaria de Hobday: media y percentiles para cada día del año.

    Para cada uno de los 366 días del año se reúnen los valores del periodo base
    de ese día y de los (`ventana` - 1) / 2 días anteriores y posteriores (de todos
    los años), y se calcula la media y los percentiles. Después cada curva se
    suaviza con una media móvil circular de `suavizado` días. Los NaN se ignoran.

    Args:
        serie (pandas.Series): Serie diaria con índice de fechas.
        base (tuple): Años de inicio y fin (incluidos) del periodo base.
        ventana (int): Ancho (impar) de la ventana de días que se junta, 11 por defecto.
        percentiles (tuple): Percentiles a calcular. 90 para calor y 10 para frío.
        suavizado (int): Ancho (impar) de la media móvil de suavizado, 31 por defecto.

    Returns:
        pandas.DataFrame: Índice `dia` (1 a 366, ver `dia_del_anio_366`) y columnas
        `media` y una por percentil (`p90`, `p10`).
    """
    if ventana % 2 == 0 or suavizado % 2 == 0:
        raise ValueError("ventana y suavizado deben ser impares.")
    en_base = (serie.index.year >= base[0]) & (serie.index.year <= base[1])
    valores = serie.to_numpy(dtype=float)[en_base]
    dia = dia_del_anio_366(serie.index[en_base])
    if np.isfinite(valores).sum() == 0:
        raise ValueError(f"La serie no tiene datos entre {base[0]} y {base[1]}.")

    columnas = {"media": np.full(366, np.nan)}
    columnas.update({f"p{p}": np.full(366, np.nan) for p in percentiles})
    for d in range(1, 367):
        distancia = (dia - d + 183) % 366 - 183  # distancia circular entre días del año
        junto = valores[np.abs(distancia) <= ventana // 2]
        junto = junto[np.isfinite(junto)]
        if junto.size == 0:
            continue
        columnas["media"][d - 1] = junto.mean()
        for p in percentiles:
            columnas[f"p{p}"][d - 1] = np.percentile(junto, p)
    suave = {nombre: _media_movil_circular(curva, suavizado) for nombre, curva in columnas.items()}
    return pd.DataFrame(suave, index=pd.RangeIndex(1, 367, name="dia"))


def aplicar_climatologia(serie, clima):
    """
    Pone al lado de cada valor de la serie su media y sus umbrales del día del año.

    Args:
        serie (pandas.Series): Serie diaria.
        clima (pandas.DataFrame): Resultado de `climatologia_hobday`.

    Returns:
        pandas.DataFrame: Mismo índice que `serie`, con la columna `valor` y las de
        `clima` (`media`, `p90`, `p10`) alineadas por día del año.
    """
    dia = dia_del_anio_366(serie.index)
    tabla = clima.loc[dia].set_index(serie.index)
    tabla.insert(0, "valor", serie.to_numpy(dtype=float))
    return tabla


# ---------------------------------------------------------------------------
# Detección de eventos
# ---------------------------------------------------------------------------
def _rachas(excede):
    """Inicio y fin (posiciones, ambos incluidos) de cada racha de True."""
    borde = np.diff(np.concatenate([[0], excede.astype(np.int8), [0]]))
    return np.flatnonzero(borde == 1), np.flatnonzero(borde == -1) - 1


def _rachas_de_eventos(excede, sin_dato, duracion_min, brecha_max):
    """
    Pasos de Hobday sobre un vector de excedencias: primero descartar, después unir.

    1. Rachas de excedencias de `duracion_min` días o más.
    2. Dos rachas separadas por `brecha_max` días o menos se unen, salvo que en la
       brecha haya algún día sin dato (nunca se cruza un vacío).
    """
    inicios, finales = _rachas(excede)
    largas = (finales - inicios + 1) >= duracion_min
    inicios, finales = inicios[largas], finales[largas]

    vacios_antes = np.concatenate([[0], np.cumsum(sin_dato)])  # vacíos en posiciones [0, i)
    unidos = []
    for ini, fin in zip(inicios, finales):
        if unidos:
            anterior_fin = unidos[-1][1]
            brecha = ini - anterior_fin - 1
            if brecha <= brecha_max and vacios_antes[ini] - vacios_antes[anterior_fin + 1] == 0:
                unidos[-1][1] = fin
                continue
        unidos.append([ini, fin])
    return unidos


def detectar_eventos(serie, clima, tipo="calor", duracion_min=5, brecha_max=2, columna_umbral=None):
    """
    Detecta olas de calor o de frío marinas con el método de Hobday.

    Un día excede si `serie` > umbral (calor, por defecto el p90) o < umbral (frío,
    por defecto el p10). Un evento es una racha de al menos `duracion_min` días
    seguidos que exceden; las rachas separadas por `brecha_max` días o menos se
    unen en un solo evento (los días de la brecha forman parte de él). Primero se
    descartan las rachas cortas y después se unen. Un día sin dato (NaN) no
    excede: corta las rachas y una unión no puede cruzarlo. Si la serie no es
    diaria y completa, se completa con NaN (`asfreq("D")`).

    Args:
        serie (pandas.Series): Serie diaria con índice de fechas.
        clima (pandas.DataFrame): Resultado de `climatologia_hobday` (debe tener
            `media` y la columna del umbral).
        tipo (str): "calor" o "frio".
        duracion_min (int): Duración mínima de una racha, en días.
        brecha_max (int): Brecha máxima, en días, para unir dos eventos.
        columna_umbral (str, optional): Columna de `clima` con el umbral. Por
            defecto "p90" para calor y "p10" para frío.

    Returns:
        pandas.DataFrame: Un evento por fila, con `tipo`, `inicio`, `fin`,
        `duracion` (días, incluye la brecha si se unieron), `dia_pico` (fecha de la
        mayor anomalía), `intensidad_max`, `intensidad_media` e
        `intensidad_acumulada` (anomalías respecto de la media climatológica,
        en °C, °C y °C-día; negativas en frío), `categoria` (1 a 4 según cuántas
        veces la anomalía del pico supera la distancia entre la media y el umbral:
        Hobday et al. 2018) y `en_curso` (True si el evento llega al último día de
        la serie). Vacía (sin filas) si no hay eventos.
    """
    if tipo not in ("calor", "frio"):
        raise ValueError("tipo debe ser 'calor' o 'frio'.")
    columna_umbral = columna_umbral or ("p90" if tipo == "calor" else "p10")
    serie = serie.asfreq("D")
    tabla = aplicar_climatologia(serie, clima)
    valor, umbral = tabla["valor"].to_numpy(), tabla[columna_umbral].to_numpy()
    anomalia = valor - tabla["media"].to_numpy()
    sin_dato = ~np.isfinite(valor)
    with np.errstate(invalid="ignore"):  # comparar con NaN da False: un vacío no excede
        excede = valor > umbral if tipo == "calor" else valor < umbral

    filas = []
    for ini, fin in _rachas_de_eventos(excede, sin_dato, duracion_min, brecha_max):
        tramo = slice(ini, fin + 1)
        pico = ini + (np.argmax(anomalia[tramo]) if tipo == "calor" else np.argmin(anomalia[tramo]))
        margen = umbral[pico] - tabla["media"].iloc[pico]  # distancia media-umbral el día del pico
        categoria = int(np.clip(np.floor(anomalia[pico] / margen), 1, 4)) if margen else 1
        filas.append({
            "tipo": tipo,
            "inicio": serie.index[ini],
            "fin": serie.index[fin],
            "duracion": fin - ini + 1,
            "dia_pico": serie.index[pico],
            "intensidad_max": anomalia[pico],
            "intensidad_media": anomalia[tramo].mean(),
            "intensidad_acumulada": anomalia[tramo].sum(),
            "categoria": categoria,
            "en_curso": bool(fin == len(serie) - 1),
        })
    columnas = ["tipo", "inicio", "fin", "duracion", "dia_pico", "intensidad_max", "intensidad_media",
                "intensidad_acumulada", "categoria", "en_curso"]
    return pd.DataFrame(filas, columns=columnas)


def marcar_dias_evento(serie, eventos):
    """
    Serie booleana (mismo índice diario que `serie`): True en los días dentro de un evento.

    Sirve para dibujar los eventos o compararlos con otro método
    (ver `solapamiento`).
    """
    marca = pd.Series(False, index=serie.asfreq("D").index)
    for evento in eventos.itertuples():
        marca.loc[evento.inicio:evento.fin] = True
    return marca


def resumen_eventos(eventos, por=None):
    """
    Cuenta y describe los eventos, en total o por década o por año.

    Args:
        eventos (pandas.DataFrame): Resultado de `detectar_eventos`.
        por (str, optional): None (una sola fila), "decada" o "anio" (según la fecha
            de inicio del evento).

    Returns:
        pandas.DataFrame: Con `n_eventos`, `dias_totales`, `duracion_media`,
        `duracion_max` e `intensidad_max` (la más extrema, en valor absoluto, con su signo).
    """
    if eventos.empty:
        return pd.DataFrame({"n_eventos": [0]}) if por is None else pd.DataFrame()
    if por is None:
        grupo = np.zeros(len(eventos), dtype=int)
    elif por == "decada":
        grupo = eventos["inicio"].dt.year // 10 * 10
    elif por == "anio":
        grupo = eventos["inicio"].dt.year
    else:
        raise ValueError("por debe ser None, 'decada' o 'anio'.")
    tabla = eventos.assign(grupo=np.asarray(grupo)).groupby("grupo")
    resumen = pd.DataFrame({
        "n_eventos": tabla.size(),
        "dias_totales": tabla["duracion"].sum(),
        "duracion_media": tabla["duracion"].mean().round(1),
        "duracion_max": tabla["duracion"].max(),
        "intensidad_max": tabla["intensidad_max"].agg(lambda x: x.loc[x.abs().idxmax()]).round(2),
    })
    resumen.index.name = {"decada": "decada", "anio": "anio"}.get(por)
    return resumen.reset_index(drop=True) if por is None else resumen


def evento_en_curso(eventos):
    """
    El evento que sigue activo en el último día de la serie, o None si no hay.

    Args:
        eventos (pandas.DataFrame): Resultado de `detectar_eventos`.

    Returns:
        pandas.Series o None: La fila del evento en curso (`inicio`, `duracion`, ...).
    """
    activos = eventos[eventos["en_curso"]]
    return None if activos.empty else activos.iloc[-1]


# ---------------------------------------------------------------------------
# Malla
# ---------------------------------------------------------------------------
def frecuencia_eventos(malla, tipo="calor", base=(1991, 2020), periodo=None, **kwargs):
    """
    Cuenta los eventos de cada píxel de una malla (el mismo método de Hobday, píxel a píxel).

    La climatología se calcula siempre con `base`; después se cuentan los
    eventos que empiezan dentro de `periodo`. Los píxeles sin datos (tierra)
    quedan NaN. Un píxel tarda unas décimas de segundo; una malla de 24 x 20,
    unos 30 segundos.

    Args:
        malla (xarray.DataArray): Temperatura con dimensiones (time, lat, lon),
            por ejemplo una malla diaria de temperatura del mar.
        tipo (str): "calor" o "frio".
        base (tuple): Años del periodo base de la climatología.
        periodo (tuple, optional): Años (inicio, fin), incluidos, en que se cuentan
            los eventos. Por defecto toda la serie.
        **kwargs: Se pasan a `detectar_eventos` (`duracion_min`, `brecha_max`).

    Returns:
        xarray.Dataset: Con `n_eventos`, `dias_en_evento`, `dias_con_dato` (días con
        dato en el periodo) y `eventos_por_decada` (n_eventos normalizado por los
        días con dato, para que un vacío no reste eventos), todos sobre (lat, lon).
    """
    import xarray as xr

    forma = (malla.sizes["lat"], malla.sizes["lon"])
    n_eventos, dias_evento, dias_dato = (np.full(forma, np.nan) for _ in range(3))
    for i in range(forma[0]):
        for j in range(forma[1]):
            serie = malla.isel(lat=i, lon=j).to_series()
            if serie.notna().sum() == 0:
                continue
            clima = climatologia_hobday(serie, base=base)
            eventos = detectar_eventos(serie, clima, tipo, **kwargs)
            if periodo is not None:
                eventos = eventos[eventos["inicio"].dt.year.between(*periodo)]
                serie = serie[(serie.index.year >= periodo[0]) & (serie.index.year <= periodo[1])]
            n_eventos[i, j] = len(eventos)
            dias_evento[i, j] = eventos["duracion"].sum()
            dias_dato[i, j] = serie.notna().sum()
    coords = {"lat": malla["lat"], "lon": malla["lon"]}
    salida = xr.Dataset({
        "n_eventos": (("lat", "lon"), n_eventos),
        "dias_en_evento": (("lat", "lon"), dias_evento),
        "dias_con_dato": (("lat", "lon"), dias_dato),
    }, coords=coords)
    salida["eventos_por_decada"] = salida["n_eventos"] / (salida["dias_con_dato"] / 3652.5)
    salida["eventos_por_decada"].attrs = {"long_name": f"eventos de {tipo} por década", "units": "eventos/década"}
    return salida


# ---------------------------------------------------------------------------
# Aprendizaje automático: Isolation Forest
# ---------------------------------------------------------------------------
def aislamiento_forest(serie, clima=None, contaminacion=0.1, dias=7, arboles=200, semilla=0):
    """
    Marca días atípicos con Isolation Forest (sin umbrales climatológicos).

    Isolation Forest aísla los puntos raros con árboles de decisión aleatorios:
    los atípicos se aíslan en pocos cortes. Aquí cada día se describe con tres
    características: la anomalía respecto de la media climatológica, esa anomalía
    promediada en `dias` días y el cambio de la anomalía en `dias` días. Los días
    sin dato (NaN) no se usan y quedan como no atípicos.

    A diferencia de Hobday, el método no exige duración mínima ni sabe del
    percentil: hay que fijar `contaminacion`, la fracción de días que se espera
    atípica (con 0,1 marca el 10 % más raro, sea calor o frío).

    Args:
        serie (pandas.Series): Serie diaria.
        clima (pandas.DataFrame, optional): Resultado de `climatologia_hobday`. Si
            es None se calcula con el periodo base por defecto.
        contaminacion (float): Fracción esperada de días atípicos (0 a 0,5).
        dias (int): Ancho de las características móviles, en días.
        arboles (int): Número de árboles del bosque.
        semilla (int): Semilla aleatoria (resultado reproducible).

    Returns:
        pandas.DataFrame: Índice diario, con `anomalia`, `anomalia_movil`,
        `cambio`, `puntaje` (menor = más atípico), `atipico`, `atipico_calor`
        (atípico con anomalía positiva) y `atipico_frio`.
    """
    from sklearn.ensemble import IsolationForest

    serie = serie.asfreq("D")
    clima = climatologia_hobday(serie) if clima is None else clima
    tabla = aplicar_climatologia(serie, clima)
    anomalia = tabla["valor"] - tabla["media"]
    caracteristicas = pd.DataFrame({
        "anomalia": anomalia,
        "anomalia_movil": anomalia.rolling(dias, center=True, min_periods=(dias + 1) // 2).mean(),
        "cambio": anomalia.diff(dias),  # NaN si falta alguno de los dos días: no cruza vacíos
    })
    completas = caracteristicas.dropna()
    bosque = IsolationForest(n_estimators=arboles, contamination=contaminacion, random_state=semilla)
    bosque.fit(completas)

    salida = caracteristicas.copy()
    salida["puntaje"] = np.nan
    salida.loc[completas.index, "puntaje"] = bosque.score_samples(completas)
    salida["atipico"] = False
    salida.loc[completas.index, "atipico"] = bosque.predict(completas) == -1
    salida["atipico_calor"] = salida["atipico"] & (salida["anomalia"] > 0)
    salida["atipico_frio"] = salida["atipico"] & (salida["anomalia"] < 0)
    return salida


def solapamiento(a, b):
    """
    Compara dos marcas de días (booleanas, mismo índice): cuántos días comparten.

    Args:
        a, b (pandas.Series): Booleanas, por ejemplo `marcar_dias_evento(...)` y
            `aislamiento_forest(...)["atipico_calor"]`.

    Returns:
        dict: `dias_a`, `dias_b`, `comunes` y `jaccard` (comunes / unión, de 0 a 1).
    """
    a, b = a.astype(bool), b.reindex(a.index, fill_value=False).astype(bool)
    comunes, union = int((a & b).sum()), int((a | b).sum())
    return {"dias_a": int(a.sum()), "dias_b": int(b.sum()), "comunes": comunes,
            "jaccard": comunes / union if union else np.nan}


# ---------------------------------------------------------------------------
# Series irregulares (Landsat 8 y 9): climatología por ventana, anomalías y episodios
# ---------------------------------------------------------------------------
def climatologia_ventana(serie, base=(2013, 2024), ventana=31, percentiles=(90, 10), suavizado=15, minimo_obs=8):
    """
    Climatología de una serie irregular: media y percentiles por día del año.

    Es la idea de `climatologia_hobday` adaptada a pocas observaciones: para cada uno de
    los 366 días del año se juntan **las observaciones válidas del periodo base que caen a
    menos de (`ventana` - 1) / 2 días de ese día, de todos los años**, y se calcula la media
    y los percentiles. Con Landsat (una pasada cada 8 días, y muchas con nubes) una
    ventana de 31 días junta de 10 a 30 observaciones; con la de 11 días de Hobday no
    alcanzan. Después cada curva se suaviza con una media móvil circular.

    Args:
        serie (pandas.Series): Serie con índice de fechas (puede traer NaN y fechas irregulares).
        base (tuple): Años de inicio y fin (incluidos) del periodo base. Por defecto
            2013-2024: los años completos de Landsat 8 (con Landsat 9 desde 2022).
        ventana (int): Ancho (impar) de la ventana de días, 31 por defecto.
        percentiles (tuple): Percentiles a calcular. 90 para calor y 10 para frío.
        suavizado (int o None): Ancho (impar) de la media móvil de suavizado de las curvas
            (15 por defecto) o None para no suavizar.
        minimo_obs (int): Observaciones mínimas en la ventana; con menos, ese día del año
            queda en NaN (no se inventa una climatología).

    Returns:
        pandas.DataFrame: Índice `dia` (1 a 366, ver `dia_del_anio_366`) y columnas `media`,
        una por percentil (`p90`, `p10`) y `n` (observaciones juntadas, antes de suavizar).
    """
    if ventana % 2 == 0 or (suavizado is not None and suavizado % 2 == 0):
        raise ValueError("ventana y suavizado deben ser impares.")
    serie = serie.dropna()
    serie = serie[(serie.index.year >= base[0]) & (serie.index.year <= base[1])]
    if serie.empty:
        raise ValueError(f"La serie no tiene datos entre {base[0]} y {base[1]}.")
    valores, dia = serie.to_numpy(dtype=float), dia_del_anio_366(serie.index)
    columnas = {"media": np.full(366, np.nan)}
    columnas.update({f"p{p}": np.full(366, np.nan) for p in percentiles})
    cuenta = np.zeros(366, dtype=int)
    for d in range(1, 367):
        distancia = (dia - d + 183) % 366 - 183  # distancia circular entre días del año
        junto = valores[np.abs(distancia) <= ventana // 2]
        cuenta[d - 1] = junto.size
        if junto.size < minimo_obs:
            continue
        columnas["media"][d - 1] = junto.mean()
        for p in percentiles:
            columnas[f"p{p}"][d - 1] = np.percentile(junto, p)
    if suavizado:
        columnas = {nombre: _suavizar_con_huecos(curva, suavizado) for nombre, curva in columnas.items()}
    return pd.DataFrame({**columnas, "n": cuenta}, index=pd.RangeIndex(1, 367, name="dia"))


def _suavizar_con_huecos(curva, ancho):
    """Media móvil circular que ignora los NaN (un día sin climatología no contamina a los vecinos)."""
    mitad = ancho // 2
    ampliada = np.concatenate([curva[-mitad:], curva, curva[:mitad]])
    validos = np.isfinite(ampliada)
    suma = np.convolve(np.where(validos, ampliada, 0.0), np.ones(ancho), mode="valid")
    n = np.convolve(validos.astype(float), np.ones(ancho), mode="valid")
    with np.errstate(invalid="ignore"):
        suave = suma / n
    return np.where(np.isfinite(curva), suave, np.nan)  # no se rellenan los días sin climatología


def anomalias_observaciones(serie, clima, percentiles=(90, 10)):
    """
    Pone al lado de cada observación su media y sus umbrales del día del año, y su anomalía.

    Args:
        serie (pandas.Series): Serie con índice de fechas (con NaN donde hay nubes).
        clima (pandas.DataFrame): Resultado de `climatologia_ventana`.
        percentiles (tuple): (alto, bajo) que existen en `clima`: (90, 10) por defecto.

    Returns:
        pandas.DataFrame: Mismo índice que `serie`: `valor`, `media`, `p90`, `p10`,
        `anomalia` (valor menos media), `excede_calor` (sobre el percentil alto) y
        `excede_frio` (bajo el bajo). Una fecha sin dato no excede nada.
    """
    alto, bajo = f"p{percentiles[0]}", f"p{percentiles[1]}"
    dia = dia_del_anio_366(serie.index)
    tabla = clima.loc[dia, ["media", alto, bajo]].set_index(serie.index)
    tabla.insert(0, "valor", serie.to_numpy(dtype=float))
    tabla["anomalia"] = tabla["valor"] - tabla["media"]
    with np.errstate(invalid="ignore"):
        tabla["excede_calor"] = tabla["valor"] > tabla[alto]
        tabla["excede_frio"] = tabla["valor"] < tabla[bajo]
    return tabla


def detectar_episodios(serie, clima, tipo="calor", min_obs=2, brecha_max_dias=24, columna_umbral=None):
    """
    Episodios de observaciones anómalas seguidas en una serie irregular.

    Es la versión "por observaciones" de `detectar_eventos` (Hobday). Un **episodio** es una
    racha de **al menos `min_obs` observaciones válidas consecutivas** que superan el
    percentil 90 (calor) o quedan bajo el 10 (frío) de su día del año. Reglas:

    * Las fechas con nubes (NaN) no cuentan como observación: ni cortan ni alargan la racha.
    * Entre dos observaciones consecutivas del episodio pueden pasar como máximo
      `brecha_max_dias` días (con 8 días de revisita, 24 es tres pasadas): una racha **no
      cruza un hueco largo**, porque no sabemos qué pasó ahí.
    * Una observación válida que no excede (un día normal en medio) corta la racha.

    **No es un evento de Hobday**: no se sabe qué pasó entre dos pasadas, así que la
    `duracion` (días entre la primera y la última observación del episodio, ambos incluidos)
    es solo una cota mínima, y no hay unión de brechas ni categorías.

    Args:
        serie (pandas.Series): Serie con índice de fechas (con NaN donde hay nubes).
        clima (pandas.DataFrame): Resultado de `climatologia_ventana`.
        tipo (str): "calor" o "frio".
        min_obs (int): Observaciones válidas consecutivas mínimas.
        brecha_max_dias (int): Días máximos entre dos observaciones consecutivas del episodio.
        columna_umbral (str, optional): Columna de `clima` a usar como umbral ("p90" o "p10" por defecto).

    Returns:
        pandas.DataFrame: Una fila por episodio: `tipo`, `inicio`, `fin`, `duracion` (días),
        `n_obs`, `dia_pico`, `intensidad_max` (anomalía del pico, °C; negativa en frío),
        `intensidad_media` y `en_curso` (True si termina en la última observación válida de
        la serie). Vacía (sin filas) si no hay episodios.
    """
    if tipo not in ("calor", "frio"):
        raise ValueError("tipo debe ser 'calor' o 'frio'.")
    columna_umbral = columna_umbral or ("p90" if tipo == "calor" else "p10")
    serie = serie.dropna()
    dia = dia_del_anio_366(serie.index)
    tabla = pd.DataFrame({"valor": serie.to_numpy(dtype=float),
                          "anomalia": serie.to_numpy(dtype=float) - clima["media"].loc[dia].to_numpy(),
                          "umbral": clima[columna_umbral].loc[dia].to_numpy()}, index=serie.index)
    tabla = tabla.dropna(subset=["umbral"])  # un día sin climatología no se puede evaluar: no cuenta como observación normal
    columnas = ["tipo", "inicio", "fin", "duracion", "n_obs", "dia_pico", "intensidad_max", "intensidad_media", "en_curso"]
    if tabla.empty:
        return pd.DataFrame(columns=columnas)
    with np.errstate(invalid="ignore"):
        excede = (tabla["valor"] > tabla["umbral"]) if tipo == "calor" else (tabla["valor"] < tabla["umbral"])
    excede = excede.to_numpy()
    fechas = tabla.index
    saltos = np.diff(fechas.to_numpy()).astype("timedelta64[D]").astype(int)
    filas = []
    inicio = None
    for i in range(len(tabla)):
        sigue = excede[i] and inicio is not None and saltos[i - 1] <= brecha_max_dias
        if excede[i] and not sigue:
            inicio = i
        elif not excede[i]:
            inicio = None
        if inicio is not None and (i == len(tabla) - 1 or not (excede[i + 1] and saltos[i] <= brecha_max_dias)):
            if i - inicio + 1 >= min_obs:
                tramo = slice(inicio, i + 1)
                anomalia = tabla["anomalia"].to_numpy()[tramo]
                pico = inicio + (np.argmax(anomalia) if tipo == "calor" else np.argmin(anomalia))
                filas.append({
                    "tipo": tipo, "inicio": fechas[inicio], "fin": fechas[i],
                    "duracion": (fechas[i] - fechas[inicio]).days + 1, "n_obs": i - inicio + 1,
                    "dia_pico": fechas[pico], "intensidad_max": tabla["anomalia"].iloc[pico],
                    "intensidad_media": anomalia.mean(), "en_curso": bool(i == len(tabla) - 1),
                })
            inicio = None
    return pd.DataFrame(filas, columns=columnas)


TIPOS_AGUA = ("bahia", "pluma", "golfo", "laguna")


def controlar_calidad_agua(tabla, rojo_max=0.05, swir1_max=0.02, tipos=TIPOS_AGUA):
    """
    Descarta las observaciones de puntos de agua que la reflectancia delata como nube o neblina.

    La máscara `qa_pixel` de Landsat marca las nubes de confianza alta; una nube delgada o la
    neblina costera pueden quedar sin marcar y se miden como "agua muy fría" (hasta 10 a 15 °C por
    debajo de lo normal). Pero el agua abierta tiene una firma clara: reflectancia roja baja (menos de
    0,05) y SWIR1 casi nula (menos de 0,02). Una observación sobre agua con reflectancia roja
    mayor que `rojo_max` o con SWIR1 mayor que `swir1_max` es sospechosa y se deja en NaN **en todas
    sus variables** (la fecha se conserva). Se usa el brillo y no el NDWI: sobre agua oscura la
    reflectancia NIR es a menudo ligeramente negativa, `convertir_a_reflectancia` la pasa a NaN y el
    NDWI no se puede calcular (en 17 a 25 % de las observaciones de agua abierta).

    Los puntos de tierra (por ejemplo `humedal`) no se tocan: su reflectancia y su NDWI son otros.

    Args:
        tabla (pandas.DataFrame): Resultado de `cargar_landsat("puntos")`.
        rojo_max (float): Reflectancia roja máxima del agua (0 a 1).
        swir1_max (float): Reflectancia SWIR1 máxima del agua (0 a 1).
        tipos (iterable): Valores de la columna `tipo` que son agua.

    Returns:
        pandas.DataFrame: Una copia de `tabla` con las variables en NaN en las observaciones sospechosas.
    """
    columnas = ["st_dn", "st_c", "verde", "rojo", "nir", "swir1", "ndwi", "ndvi"]
    limpia = tabla.copy()
    sospechosa = limpia["tipo"].isin(tipos) & ((limpia["rojo"] > rojo_max) | (limpia["swir1"] > swir1_max))
    limpia.loc[sospechosa, columnas] = np.nan
    return limpia


def episodios_por_punto(tabla, variable="st_c", tipos=None, **kwargs):
    """
    Episodios cálidos y fríos de cada punto de la tabla, en una sola tabla.

    Para cada punto: `serie_landsat`, `climatologia_ventana` y `detectar_episodios` de calor y de frío.

    Args:
        tabla (pandas.DataFrame): Resultado de `cargar_landsat("puntos")` (conviene pasarla antes por
            `controlar_calidad_agua`).
        variable (str): Variable a analizar, "st_c" por defecto.
        tipos (iterable, optional): Solo los puntos con estos valores de `tipo`. Por defecto todos.
        **kwargs: Se pasan a `detectar_episodios` (`min_obs`, `brecha_max_dias`).

    Returns:
        pandas.DataFrame: Las columnas de `detectar_episodios` más `punto` (el `id`) y `nombre`.
    """
    nombres = tabla.drop_duplicates("id").set_index("id")["nombre"]
    ids = tabla.drop_duplicates("id")
    if tipos is not None:
        ids = ids[ids["tipo"].isin(tipos)]
    partes = []
    for punto in ids["id"]:
        serie = serie_landsat(tabla, punto, variable)
        clima = climatologia_ventana(serie)
        for tipo in ("calor", "frio"):
            ev = detectar_episodios(serie, clima, tipo, **kwargs)
            if not ev.empty:
                partes.append(ev.assign(punto=punto, nombre=nombres[punto]))
    columnas = ["punto", "nombre", "tipo", "inicio", "fin", "duracion", "n_obs", "dia_pico", "intensidad_max",
                "intensidad_media", "en_curso"]
    todo = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=columnas)
    return todo[columnas].sort_values(["punto", "inicio"]).reset_index(drop=True)


def aislamiento_forest_observaciones(tabla, columnas, clima_por_variable=None, contaminacion=0.1, arboles=200, semilla=0):
    """
    Isolation Forest sobre varias variables de las observaciones válidas de un punto.

    Cada observación se describe con la **anomalía** de cada variable respecto de su
    climatología (si se pasa `clima_por_variable`) o con el valor tal cual. Es el análisis
    multivariable que permite Landsat: la temperatura de superficie y la reflectancia
    medidas **en el mismo instante**. Las fechas con algún NaN no se usan y quedan como no
    atípicas. No hay umbral ni noción de duración: hay que fijar `contaminacion`, la
    fracción de observaciones que se espera atípica.

    Args:
        tabla (pandas.DataFrame): Índice de fechas y una columna por variable (por ejemplo
            `st_c`, `rojo`, `ndwi`), con NaN donde hay nubes.
        columnas (list): Variables a usar.
        clima_por_variable (dict, optional): `{variable: climatología de climatologia_ventana}`.
            Si se indica, cada variable se reemplaza por su anomalía.
        contaminacion (float): Fracción esperada de observaciones atípicas (0 a 0,5).
        arboles (int): Número de árboles del bosque.
        semilla (int): Semilla aleatoria (resultado reproducible).

    Returns:
        pandas.DataFrame: Mismo índice que `tabla`, con las variables usadas (o sus anomalías),
        `puntaje` (menor = más atípico) y `atipico` (bool).
    """
    from sklearn.ensemble import IsolationForest

    variables = pd.DataFrame(index=tabla.index)
    for nombre in columnas:
        serie = tabla[nombre].astype(float)
        if clima_por_variable and nombre in clima_por_variable:
            variables[nombre] = anomalias_observaciones(serie, clima_por_variable[nombre])["anomalia"]
        else:
            variables[nombre] = serie
    completas = variables.dropna()
    bosque = IsolationForest(n_estimators=arboles, contamination=contaminacion, random_state=semilla)
    bosque.fit(completas)
    salida = variables.copy()
    salida["puntaje"] = np.nan
    salida.loc[completas.index, "puntaje"] = bosque.score_samples(completas)
    salida["atipico"] = False
    salida.loc[completas.index, "atipico"] = bosque.predict(completas) == -1
    return salida


def climatologia_malla(malla, base=(2013, 2024), ventana=31, paso=8, percentiles=(90, 10), minimo_obs=8):
    """
    Climatología por ventana de días del año en cada píxel de una malla de observaciones irregulares.

    Como `climatologia_ventana`, pero para todos los píxeles a la vez y evaluada solo cada
    `paso` días del año (la curva es suave, no hace falta cada día). Los percentiles
    ignoran los NaN (nubes) de cada píxel por separado.

    Args:
        malla (xarray.DataArray): Dimensiones (time, y, x), con NaN donde hay nubes.
        base (tuple): Años de inicio y fin (incluidos) del periodo base.
        ventana (int): Ancho (impar) de la ventana de días.
        paso (int): Separación en días del año entre los centros donde se calcula.
        percentiles (tuple): Percentiles a calcular.
        minimo_obs (int): Observaciones mínimas por píxel y centro; con menos queda NaN.

    Returns:
        xarray.Dataset: Dimensiones (centro, y, x) con `media`, `p90`, `p10` y `n`
        (observaciones válidas). `centro` es el día del año (calendario de 366 días).
    """
    tiempo = pd.DatetimeIndex(malla["time"].values)
    en_base = (tiempo.year >= base[0]) & (tiempo.year <= base[1])
    datos = malla.isel(time=np.flatnonzero(en_base))
    dia = dia_del_anio_366(pd.DatetimeIndex(datos["time"].values))
    valores = datos.to_numpy().astype("float32")
    centros = np.arange(1, 367, paso)
    salida = {nombre: np.full((len(centros), *valores.shape[1:]), np.nan, dtype="float32")
              for nombre in ("media", *[f"p{p}" for p in percentiles])}
    cuenta = np.zeros((len(centros), *valores.shape[1:]), dtype="int16")
    for k, centro in enumerate(centros):
        distancia = (dia - centro + 183) % 366 - 183
        junto = valores[np.abs(distancia) <= ventana // 2]
        n = np.isfinite(junto).sum(axis=0)
        cuenta[k] = n
        suficiente = n >= minimo_obs
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # píxeles sin ninguna observación: "Mean of empty slice"
            media = np.nanmean(junto, axis=0)
            percentil = np.nanpercentile(junto, list(percentiles), axis=0)
        salida["media"][k] = np.where(suficiente, media, np.nan)
        for j, p in enumerate(percentiles):
            salida[f"p{p}"][k] = np.where(suficiente, percentil[j], np.nan)
    coords = {"centro": centros, "y": malla["y"].values, "x": malla["x"].values}
    return xr.Dataset({nombre: (("centro", "y", "x"), v) for nombre, v in {**salida, "n": cuenta}.items()}, coords=coords)


def _centro_mas_cercano(tiempo, centros):
    """Posición del centro de la climatología más cercano (circular) a cada fecha."""
    dia = dia_del_anio_366(pd.DatetimeIndex(tiempo))
    distancia = np.abs((dia[:, None] - np.asarray(centros)[None, :] + 183) % 366 - 183)
    return distancia.argmin(axis=1)


def anomalia_malla(malla, clima):
    """
    Anomalía de cada observación de la malla respecto de la media de su píxel y de su época del año.

    Args:
        malla (xarray.DataArray): Dimensiones (time, y, x).
        clima (xarray.Dataset): Resultado de `climatologia_malla`.

    Returns:
        xarray.DataArray: `malla` menos la media climatológica (mismas dimensiones).
    """
    posicion = _centro_mas_cercano(malla["time"].values, clima["centro"].values)
    media = clima["media"].isel(centro=xr.DataArray(posicion, dims="time")).assign_coords(time=malla["time"])
    return (malla - media).rename("anomalia")


def frecuencia_anomalias_malla(malla, clima, tipo="calor", periodo=None):
    """
    Fracción de observaciones válidas anómalas (sobre el p90 o bajo el p10) en cada píxel.

    Por construcción, en el periodo base cada píxel tiene cerca del 10 % de observaciones sobre
    su p90. Un `periodo` distinto (por ejemplo los últimos años) muestra dónde se ha calentado
    o enfriado más de lo normal: valores muy por encima de 0,10 indican más anomalías que antes.

    Args:
        malla (xarray.DataArray): Dimensiones (time, y, x).
        clima (xarray.Dataset): Resultado de `climatologia_malla`.
        tipo (str): "calor" o "frio".
        periodo (tuple, optional): Años (inicio, fin), incluidos, en que se cuenta. Por defecto todos.

    Returns:
        xarray.Dataset: Dimensiones (y, x) con `n_obs` (observaciones válidas con climatología),
        `n_anomalas` y `fraccion` (NaN donde no hay observaciones).
    """
    if tipo not in ("calor", "frio"):
        raise ValueError("tipo debe ser 'calor' o 'frio'.")
    tiempo = pd.DatetimeIndex(malla["time"].values)
    if periodo is not None:
        malla = malla.isel(time=np.flatnonzero((tiempo.year >= periodo[0]) & (tiempo.year <= periodo[1])))
    posicion = _centro_mas_cercano(malla["time"].values, clima["centro"].values)
    umbral = clima["p90" if tipo == "calor" else "p10"].isel(centro=xr.DataArray(posicion, dims="time"))
    umbral = umbral.assign_coords(time=malla["time"])
    con_dato = malla.notnull() & umbral.notnull()
    excede = (malla > umbral) if tipo == "calor" else (malla < umbral)
    n_obs, n_anomalas = con_dato.sum("time"), (excede & con_dato).sum("time")
    fraccion = (n_anomalas / n_obs.where(n_obs > 0)).astype("float32")
    return xr.Dataset({"n_obs": n_obs, "n_anomalas": n_anomalas, "fraccion": fraccion})
