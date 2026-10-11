"""
Eventos extremos de temperatura del mar: olas de calor y de frío marinas.

Implementa el método de Hobday et al. (2016), el estándar para olas de calor
marinas, con pandas y numpy legibles (la idea es que puedas leer el código):

1. `climatologia_hobday`: para cada día del año, la media y los percentiles 90 y
   10 de un periodo base (1991-2020), juntando los días a +-5 días (ventana de 11)
   y suavizando con una media móvil de 31 días.
2. `detectar_eventos`: un día "excede" si supera el p90 (calor) o queda bajo el
   p10 (frío). Un evento es una racha de excedencias de al menos 5 días. Si dos
   eventos quedan separados por 2 días o menos, se unen en uno solo. El orden
   importa: primero se descartan las rachas cortas y después se unen.
3. `resumen_eventos`, `evento_en_curso`: tablas para contar y describir.

Vacíos (NaN). Una serie con días sin dato (como OISST en Planetary Computer,
que no tiene 2024-06-20 a 2026-03-05) **nunca se rellena ni se puentea**: un día
sin dato no excede el umbral, así que corta cualquier racha, y una unión de
eventos no puede cruzar un vacío (aunque sea de un solo día). La climatología
casi no se afecta (usa 30 años de base); lo que no se puede es detectar eventos
dentro del vacío.

Validación. Con OISST (1981 a 2026, con su vacío) el resultado coincide con
`xmhw` (https://github.com/coecms/xmhw, implementación de la comunidad Hobday):
los mismos 88 eventos de calor y 111 de frío, con las mismas fechas de fin y de
inicio, salvo el primer día de la serie (xmhw no cuenta el 1981-09-01 como inicio).
Las curvas de umbral y media difieren menos de 0,007 °C (el 29 de febrero y los
bordes de la ventana se tratan de forma ligeramente distinta). Con `xmhw` el
frío se pide con `coldSpells=True` y `pctile=90` (él invierte la serie).
`utils/tests/test_extremos.py` repite la comparación si `xmhw` está instalado
(`pip install git+https://github.com/coecms/xmhw`; no está en PyPI).

Además: `frecuencia_eventos` repite el método en cada píxel de una malla, y
`aislamiento_forest` marca días atípicos con Isolation Forest (aprendizaje
automático, sin umbrales) para compararlo con Hobday.

Uso:

    from utils.extremos import (cargar_oisst, climatologia_hobday, detectar_eventos,
                                resumen_eventos)
    sst = cargar_oisst()                                  # Serie diaria (con vacíos)
    clima = climatologia_hobday(sst)                      # media, p90, p10 por día del año
    olas_calor = detectar_eventos(sst, clima, "calor")
    olas_frio = detectar_eventos(sst, clima, "frio")
"""

import numpy as np
import pandas as pd

from .datos_taller import abrir_dato

NOMBRES_CATEGORIA = {1: "I moderada", 2: "II fuerte", 3: "III severa", 4: "IV extrema"}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_oisst(tipo="punto"):
    """
    Carga la temperatura superficial del mar OISST preparada para el taller.

    Lee de `talleres/fair-series-temporales/datos/` si existe y, si no, de GitHub
    (ver `utils.datos_taller`). Los días que Planetary Computer no tiene
    (2024-06-20 a 2026-03-05 y 2026-03-07 y 2026-03-08) quedan como NaN: ni se
    rellenan ni se eliminan, así el calendario es diario y completo.

    Args:
        tipo (str): "punto" (serie del píxel frente a Concepción, -36.875, -73.375,
            desde `oisst_concepcion.csv`) o "malla" (24 x 20 píxeles de 0,25
            grados, 34-40 S y 71-76 O, desde `oisst_centro_sur.nc`).

    Returns:
        pandas.Series o xarray.DataArray: En grados Celsius, con un día por paso
        de tiempo (índice `fecha`). "punto" da una Serie llamada `sst_c`;
        "malla" da un DataArray `sst` con dimensiones (time, lat, lon), con NaN en
        tierra y en los días sin dato.
    """
    if tipo == "punto":
        tabla = pd.read_csv(abrir_dato("oisst_concepcion.csv"), parse_dates=["fecha"], index_col="fecha")
        return tabla["sst_c"].asfreq("D")
    if tipo == "malla":
        import xarray as xr

        with xr.open_dataset(abrir_dato("oisst_centro_sur.nc")) as ds:
            sst = ds["sst"].load()
        dias = pd.date_range(sst.time.values[0], sst.time.values[-1], freq="D")
        return sst.reindex(time=dias)
    raise ValueError(f"tipo debe ser 'punto' o 'malla', no {tipo!r}.")


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
            por ejemplo `cargar_oisst("malla")`.
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
