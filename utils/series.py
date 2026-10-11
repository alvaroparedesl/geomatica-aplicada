"""
Series temporales por punto: de la imagen satelital a la tabla, y de la tabla al análisis.

El módulo tiene tres partes (los imports pesados están dentro de las funciones, así
los cuadernos de exploración solo necesitan pandas):

1. **Puntos**: `cargar_puntos_csv`, `puntos_a_geodataframe`.
2. **Sentinel-2 en ventanas pequeñas** alrededor de cada punto. Cargar el área
   completa que rodea a los puntos es lento (decenas de GB virtuales); cargar solo
   una ventana de unos 300 m por punto tarda minutos y pesa unos cientos de MB.
   Los helpers **no reemplazan** el patrón nativo del curso, solo preparan sus
   pasos; `cargar_ventanas` hace, para cada punto, exactamente esto:

       bbox = bbox_punto(lon, lat, radio_m)                       # ventana de ~300 m
       items_p = items_de_punto(items, bbox)                      # ítems que tocan la ventana
       parametros = parametros_carga("sentinel-2", bandas, resolucion, crs, catalogo, bbox, chunks=CHUNKS_VENTANA)
       datos = odc.stac.load(items_p, **parametros)               # llamada nativa
       datos = renombrar_bandas(datos, catalogo)
       datos = agregar_offset(datos, items_p)                     # offset de reflectancia por fecha
       datos = agregar_nubes_escena(datos, items_p)               # % de nubes de la escena por fecha

   y `construir_series` aplica `preparar_reflectancia` (máscara SCL + reflectancia con
   el offset de cada fecha), calcula los índices y resume cada ventana con la mediana
   (`utils.raster.extraer_series_puntos(..., radio=...)`).
3. **Análisis de una serie** (`pandas.Series` con índice de fechas): `rolling_media`,
   `tendencia_lineal`, `climatologia_mensual` y `anomalias_zscore`; y el guardado en
   parquet con metadatos (`guardar_series`, `cargar_series`).

La tabla de series (formato ancho, una fila por punto y fecha) tiene las columnas
`id, nombre, tipo, lat, lon, fecha, ndwi, ndci, b04, ndvi, nubes_escena,
pixeles_validos, n_pixeles, offset_reflectancia` (ver `COLUMNAS_SERIES`).
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .datos_taller import abrir_dato

COLUMNAS_PUNTOS = ["id", "nombre", "lat", "lon", "tipo"]
# Bandas de Sentinel-2 que usan los índices de agua y de vegetación (nombres genéricos de `utils.raster`)
BANDAS_SERIES = ("green", "red", "red_edge1", "nir")
# Índice de utils.raster -> columna de la tabla de series
COLUMNAS_INDICES = {"NDWI_AGUA": "ndwi", "NDCI": "ndci", "B04": "b04", "NDVI": "ndvi"}
COLUMNAS_SERIES = ["id", "nombre", "tipo", "lat", "lon", "fecha", *COLUMNAS_INDICES.values(),
                   "nubes_escena", "pixeles_validos", "n_pixeles", "offset_reflectancia"]
METROS_POR_GRADO = 111_320.0
# Una ventana pequeña cabe en un solo bloque: la carga es perezosa (dask) y se lee en paralelo al final.
CHUNKS_VENTANA = {"x": 512, "y": 512}


# ---------------------------------------------------------------------------
# Puntos
# ---------------------------------------------------------------------------
def cargar_puntos_csv(origen="puntos_concepcion.csv"):
    """
    Lee un CSV de puntos con las columnas id, nombre, lat, lon y tipo.

    Args:
        origen (str o pathlib.Path): Ruta o URL de un CSV. Un nombre simple (sin
            carpeta) se busca en `talleres/fair-series-temporales/datos/` y, si no
            está, se descarga de GitHub (ver `utils.datos_taller`).

    Returns:
        pandas.DataFrame: Columnas `id` y `nombre` (texto), `lat` y `lon` (grados
        decimales, EPSG:4326) y `tipo` (texto), en ese orden. Si el CSV trae más
        columnas, se conservan al final.

    Raises:
        ValueError: Si faltan columnas, hay `id` repetidos o las coordenadas están
            fuera de rango (un error típico es invertir lat y lon).
    """
    origen = str(origen)
    if "/" not in origen and "\\" not in origen:
        origen = abrir_dato(origen)
    puntos = pd.read_csv(origen, dtype={"id": str, "nombre": str, "tipo": str})
    faltan = [c for c in COLUMNAS_PUNTOS if c not in puntos.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan las columnas {faltan}; debe tener {COLUMNAS_PUNTOS}.")
    if puntos["id"].duplicated().any():
        raise ValueError(f"Hay id repetidos: {sorted(puntos.loc[puntos['id'].duplicated(), 'id'])}.")
    if not (puntos["lat"].between(-90, 90).all() and puntos["lon"].between(-180, 180).all()):
        raise ValueError("lat debe estar entre -90 y 90 y lon entre -180 y 180 (¿están invertidas?).")
    otras = [c for c in puntos.columns if c not in COLUMNAS_PUNTOS]
    return puntos[COLUMNAS_PUNTOS + otras]


def puntos_a_geodataframe(puntos):
    """GeoDataFrame (EPSG:4326) con los puntos de `cargar_puntos_csv`; sirve para `.explore()` o para guardar GeoJSON."""
    import geopandas as gpd

    return gpd.GeoDataFrame(puntos, geometry=gpd.points_from_xy(puntos["lon"], puntos["lat"]), crs="EPSG:4326")


def crs_utm(lon, lat):
    """CRS UTM (EPSG) de la zona que contiene el punto, por ejemplo "EPSG:32718" para Concepción (73 O, 36,8 S)."""
    zona = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zona if lat >= 0 else 32700 + zona}"


# ---------------------------------------------------------------------------
# Sentinel-2: ventanas pequeñas alrededor de cada punto (el patrón nativo, paso a paso)
# ---------------------------------------------------------------------------
def bbox_punto(lon, lat, radio_m=150):
    """
    Rectángulo [oeste, sur, este, norte] (grados) de `radio_m` metros alrededor de un punto.

    Con el radio por defecto (150 m) la ventana mide unos 300 x 300 m: 30 x 30
    píxeles de 10 m. Es el `bbox` que se pasa a `catalogo.search` o a `odc.stac.load`.
    """
    dlat = radio_m / METROS_POR_GRADO
    dlon = radio_m / (METROS_POR_GRADO * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def bbox_puntos(puntos, radio_m=150):
    """Rectángulo [oeste, sur, este, norte] que contiene las ventanas de todos los puntos (para una sola búsqueda STAC)."""
    cajas = np.array([bbox_punto(p.lon, p.lat, radio_m) for p in puntos.itertuples()])
    return [cajas[:, 0].min(), cajas[:, 1].min(), cajas[:, 2].max(), cajas[:, 3].max()]


def items_de_punto(items, bbox):
    """Los ítems STAC cuya huella toca el `bbox` (una escena puede cubrir unos puntos y otros no)."""
    from shapely.geometry import box, shape

    ventana = box(*bbox)
    return [item for item in items if shape(item.geometry).intersects(ventana)]


def agregar_nubes_escena(datos, items):
    """
    Agrega a `datos` la coordenada `nubes_escena` (% de nubes de la escena, un valor por fecha).

    Es el `eo:cloud_cover` del catálogo: describe la **escena completa** (unos 100 x
    100 km), no tu ventana. Para una fecha formada por varios ítems (mismo día solar y
    mismo offset) se promedia. Funciona igual que `utils.stac.agregar_offset`: pasa
    los mismos `items` con que cargaste `datos`, con el `groupby` de `parametros_carga`.

    Args:
        datos (xarray.Dataset): Resultado de `odc.stac.load`.
        items (iterable): Ítems usados en la carga.

    Returns:
        xarray.Dataset: `datos` con la coordenada `nubes_escena`.

    Raises:
        ValueError: Si una fecha de `datos` no corresponde a los ítems.
    """
    from .stac import agrupar_por_dia_y_offset

    grupos = {}
    for item in items:
        marca = np.datetime64(item.datetime.replace(tzinfo=None), "ns")
        clave = agrupar_por_dia_y_offset(item)
        primera, nubes = grupos.get(clave, (marca, []))
        grupos[clave] = (min(marca, primera), [*nubes, item.properties.get("eo:cloud_cover", np.nan)])
    nubes_de_fecha = {marca: float(np.mean(nubes)) for marca, nubes in grupos.values()}
    try:
        valores = [nubes_de_fecha[marca] for marca in datos["time"].values.astype("datetime64[ns]")]
    except KeyError as error:
        raise ValueError(f"Las fechas de los datos no corresponden a los ítems (falta {error.args[0]}).") from None
    datos = datos.assign_coords(nubes_escena=("time", np.array(valores, dtype=float)))
    datos["nubes_escena"].attrs = {"long_name": "nubes de la escena (eo:cloud_cover)", "units": "%"}
    return datos


def cargar_ventanas(items, puntos, bandas=None, radio_m=150, resolucion=10, crs=None, catalogo=None,
                    trabajadores=8, reintentos=2, verbose=True):
    """
    Carga una ventana de Sentinel-2 alrededor de cada punto, con todas las fechas.

    Para cada punto hace los pasos de `bbox_punto` a `agregar_nubes_escena` que
    muestra el docstring del módulo (incluido el `odc.stac.load` nativo), y al
    final lee todos los píxeles de una vez (en paralelo). Los datos quedan en
    DN, sin enmascarar ni convertir: de eso se encarga `construir_series`.

    Args:
        items (iterable): Ítems Sentinel-2 ya pasados por `utils.stac.preparar_items`
            (sin duplicados, de un solo catálogo).
        puntos (pandas.DataFrame): Resultado de `cargar_puntos_csv`.
        bandas (list, optional): Nombres de banda de Sentinel-2 (ver `utils.raster.nombres_bandas`).
            Por defecto verde, rojo, borde rojo, NIR y SCL (`BANDAS_SERIES`).
        radio_m (float): Radio de la ventana en metros.
        resolucion (int): Tamaño de píxel en metros.
        crs (str, optional): CRS de carga. Por defecto la zona UTM de los puntos (`crs_utm`).
        catalogo (str o pystac_client.Client, optional): El mismo de la búsqueda.
        trabajadores (int): Lecturas en paralelo (hilos de dask). Muchas a la vez pueden
            hacer que el servidor rechace alguna lectura.
        reintentos (int): Veces que se repite la lectura completa si una lectura falla.
        verbose (bool): Si es True, informa cada punto.

    Returns:
        dict: `{id del punto: xarray.Dataset}` con dimensiones (time, y, x), las
        bandas en DN (float32, NaN donde no hay dato) y las coordenadas `offset_reflectancia` y `nubes_escena`.
    """
    import time as reloj

    import dask
    import odc.stac

    from .raster import nombres_bandas
    from .stac import agregar_offset, parametros_carga, renombrar_bandas

    items = list(items)
    bandas = bandas or nombres_bandas("sentinel-2", BANDAS_SERIES, con_calidad=True)
    crs = crs or crs_utm(puntos["lon"].mean(), puntos["lat"].mean())
    perezosas = {}
    for p in puntos.itertuples():
        bbox = bbox_punto(p.lon, p.lat, radio_m)
        items_p = items_de_punto(items, bbox)
        if not items_p:
            raise ValueError(f"Ninguna imagen cubre el punto {p.id!r}: revisa la búsqueda o las coordenadas.")
        parametros = parametros_carga("sentinel-2", bandas, resolucion, crs, catalogo, bbox, chunks=CHUNKS_VENTANA)
        datos = odc.stac.load(items_p, **parametros)
        datos = renombrar_bandas(datos, catalogo)
        datos = agregar_offset(datos, items_p)
        datos = agregar_nubes_escena(datos, items_p)
        perezosas[p.id] = datos
        if verbose:
            print(f"{p.id}: {len(items_p)} imágenes, {datos.sizes['time']} fechas, ventana de {datos.sizes['x']} x {datos.sizes['y']} píxeles")
    inicio = reloj.time()
    for intento in range(reintentos + 1):
        try:
            (ventanas,) = dask.compute(perezosas, scheduler="threads", num_workers=trabajadores)
            break
        except Exception as error:  # una lectura de red fallida: se repite todo (lo ya leído no se pierde)
            if intento == reintentos:
                raise
            print(f"Falló una lectura ({type(error).__name__}); reintento {intento + 1} de {reintentos}.")
    if verbose:
        peso = sum(d.nbytes for d in ventanas.values()) / 1e6
        print(f"Lectura terminada: {peso:.0f} MB en {reloj.time() - inicio:.0f} s.")
    return ventanas


def construir_series(ventanas, puntos, radio_m=150, estadistico="mediana", fraccion_minima=0.5, excluir=None):
    """
    De las ventanas a la tabla de series: máscara SCL, reflectancia, índices y resumen por punto.

    Para cada punto y fecha:

    1. `preparar_reflectancia`: los píxeles que la banda SCL marca como nube, sombra,
       cirrus, saturados o sin dato pasan a NaN, y el resto se convierte a
       reflectancia con el offset de **esa fecha** (`offset_reflectancia`).
    2. Índices: NDWI de McFeeters (verde y NIR, agua), NDCI (borde rojo y rojo,
       clorofila), reflectancia roja B04 (turbidez) y NDVI (vegetación).
    3. Resumen de la ventana: el `estadistico` (mediana) de los píxeles que están a
       `radio_m` o menos del punto y no están enmascarados.
    4. `pixeles_validos` es la fracción de esos píxeles que pasó la máscara. Si es
       menor que `fraccion_minima`, los índices de esa fecha se dejan en NaN (la
       fecha se conserva, para ver los vacíos). Las fechas sin cobertura (la
       escena no cubre la ventana: todo SCL = 0) se descartan.

    Args:
        ventanas (dict): Resultado de `cargar_ventanas`.
        puntos (pandas.DataFrame): Resultado de `cargar_puntos_csv`.
        radio_m (float): Radio de la ventana de resumen, en metros.
        estadistico (str): "mediana" (por defecto), "media", "min" o "max".
        fraccion_minima (float): Fracción mínima de píxeles válidos (0 a 1).
        excluir (iterable, optional): Clases SCL a descartar (ver `mascara_calidad_sentinel2`).

    Returns:
        pandas.DataFrame: Una fila por punto y fecha, columnas `COLUMNAS_SERIES`.
    """
    import xarray as xr

    from .raster import (calcular_indices, extraer_series_puntos, lonlat_a_crs, mascara_calidad_sentinel2,
                         preparar_reflectancia)

    tablas = []
    for p in puntos.itertuples():
        datos = ventanas[p.id]
        refl = preparar_reflectancia(datos, "sentinel-2", excluir=excluir)
        indices = calcular_indices(refl, "sentinel-2", ["NDWI_AGUA", "NDCI", "NDVI"])
        indices["B04"] = refl["B04"]
        mascaras = xr.Dataset({
            "valido": mascara_calidad_sentinel2(datos["SCL"], excluir).astype("float32"),
            "cobertura": (datos["SCL"] != 0).astype("float32"),
        })
        x, y = lonlat_a_crs(p.lon, p.lat, datos.rio.crs)  # el punto, en el CRS de la ventana
        un_punto = pd.DataFrame({"id": [p.id], "x": [x], "y": [y]})
        resumen = extraer_series_puntos(indices, un_punto, "id", radio=radio_m, estadistico=estadistico)
        fracciones = extraer_series_puntos(mascaras, un_punto, "id", radio=radio_m, estadistico="media")

        ancho = resumen.pivot(index="time", columns="indice", values="valor")[list(COLUMNAS_INDICES)]
        ancho.columns = list(COLUMNAS_INDICES.values())
        frac = fracciones.pivot(index="time", columns="indice", values="valor")
        ancho["pixeles_validos"] = frac["valido"]
        ancho["n_pixeles"] = fracciones.loc[fracciones["indice"] == "valido"].set_index("time")["n_validos"]
        ancho["nubes_escena"] = datos["nubes_escena"].to_series()
        ancho["offset_reflectancia"] = datos["offset_reflectancia"].to_series()
        ancho = ancho[frac["cobertura"] > 0]  # sin cobertura: la escena no llega a la ventana
        ancho.loc[ancho["pixeles_validos"] < fraccion_minima, list(COLUMNAS_INDICES.values())] = np.nan
        ancho.index.name = "fecha"
        tablas.append(ancho.reset_index().assign(id=p.id))

    tabla = pd.concat(tablas, ignore_index=True).merge(puntos[["id", "nombre", "tipo", "lat", "lon"]], on="id")
    return tabla[COLUMNAS_SERIES].sort_values(["id", "fecha"]).reset_index(drop=True)


def resumen_validez(tabla, variable="ndwi"):
    """
    Cuántas fechas aportan un valor útil en cada punto.

    Args:
        tabla (pandas.DataFrame): Resultado de `construir_series` o `cargar_series`.
        variable (str): Columna que debe tener valor para contar la fecha como válida.

    Returns:
        pandas.DataFrame: Por punto: `fechas` (con cobertura), `fechas_validas`
        (con `variable` no NaN), `porcentaje_valido`, primera y última fecha válida,
        y la media de `nubes_escena` y de `pixeles_validos`.
    """
    por_punto = tabla.groupby("id", sort=False)
    valida = tabla.dropna(subset=[variable]).groupby("id", sort=False)["fecha"]
    resumen = pd.DataFrame({
        "nombre": por_punto["nombre"].first(),
        "fechas": por_punto.size(),
        "fechas_validas": valida.size(),
        "primera_valida": valida.min(),
        "ultima_valida": valida.max(),
        "nubes_escena_media": por_punto["nubes_escena"].mean().round(1),
        "pixeles_validos_medio": por_punto["pixeles_validos"].mean().round(2),
    }).fillna({"fechas_validas": 0})
    resumen["fechas_validas"] = resumen["fechas_validas"].astype(int)
    resumen.insert(3, "porcentaje_valido", (100 * resumen["fechas_validas"] / resumen["fechas"]).round(0))
    return resumen


# ---------------------------------------------------------------------------
# Guardar y cargar (parquet)
# ---------------------------------------------------------------------------
def guardar_series(tabla, ruta, metadatos=None):
    """
    Guarda la tabla de series en parquet, con metadatos dentro del mismo archivo.

    Parquet conserva los tipos (fechas, números) y pesa poco. Los metadatos (título,
    fuente, licencia, descripción de columnas...) quedan en el archivo y se leen con
    `leer_metadatos_series`: es la parte "documentada" de un dato FAIR.

    Args:
        tabla (pandas.DataFrame): Por ejemplo el resultado de `construir_series`.
        ruta (str o pathlib.Path): Archivo `.parquet` de salida.
        metadatos (dict, optional): Texto o diccionarios (se guardan como JSON).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    tabla_pa = pa.Table.from_pandas(tabla, preserve_index=False)
    meta = dict(tabla_pa.schema.metadata or {})
    for clave, valor in (metadatos or {}).items():
        texto = valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False)
        meta[clave.encode()] = texto.encode()
    pq.write_table(tabla_pa.replace_schema_metadata(meta), ruta, compression="zstd")


def _ruta_series(origen):
    origen = Path(origen)
    return origen if origen.exists() or origen.parent != Path(".") else abrir_dato(origen.name)


def cargar_series(origen="series_sentinel2_concepcion.parquet"):
    """
    Lee un parquet de series (`guardar_series`) como DataFrame.

    Args:
        origen (str o pathlib.Path): Ruta del archivo. Un nombre simple se busca en
            `talleres/fair-series-temporales/datos/` y, si no está, en GitHub.

    Returns:
        pandas.DataFrame: Con los tipos originales (`fecha` es datetime).
    """
    return pd.read_parquet(_ruta_series(origen))


def leer_metadatos_series(origen="series_sentinel2_concepcion.parquet"):
    """Metadatos guardados en un parquet por `guardar_series`, como diccionario de texto."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(_ruta_series(origen)).metadata or {}
    return {k.decode(): v.decode() for k, v in meta.items() if k != b"pandas"}


# ---------------------------------------------------------------------------
# Análisis de una serie (pandas.Series con índice de fechas)
# ---------------------------------------------------------------------------
def serie_de_punto(tabla, punto, variable, columna_id="id"):
    """
    La serie de una variable en un punto: `pandas.Series` con índice `fecha`, sin NaN.

    Args:
        tabla (pandas.DataFrame): Tabla de series (`cargar_series`).
        punto (str): Valor de la columna `id`.
        variable (str): Columna, por ejemplo "ndwi".
    """
    sub = tabla[tabla[columna_id] == punto].dropna(subset=[variable]).sort_values("fecha")
    return pd.Series(sub[variable].to_numpy(), index=pd.DatetimeIndex(sub["fecha"], name="fecha"), name=variable)


def rolling_media(serie, ventana="30D", minimo=3, centrada=True):
    """
    Media móvil de una serie con fechas irregulares (las de Sentinel-2 no son diarias).

    Args:
        serie (pandas.Series): Índice de fechas.
        ventana (str o int): Ancho de la ventana. Un texto como "30D" son días de
            calendario (adecuado para series irregulares); un entero son observaciones.
        minimo (int): Observaciones mínimas dentro de la ventana; si hay menos, NaN.
        centrada (bool): Si es True la ventana queda centrada en cada fecha.

    Returns:
        pandas.Series: Misma longitud e índice que `serie`.
    """
    return serie.sort_index().rolling(ventana, min_periods=minimo, center=centrada).mean()


class Tendencia(NamedTuple):
    """Resultado de `tendencia_lineal`."""

    pendiente_anual: float  # unidades de la serie por año
    intercepto: float  # valor ajustado en la primera fecha
    r2: float
    p_valor: float
    n: int
    ajuste: pd.Series  # la recta ajustada, con el índice de la serie


def tendencia_lineal(serie):
    """
    Tendencia lineal (mínimos cuadrados) de una serie respecto del tiempo.

    El tiempo se mide en años desde la primera fecha, así que la pendiente es el
    cambio por año. El valor p es el de la pendiente (prueba t); recuerda que supone
    observaciones independientes, algo que una serie temporal casi nunca cumple del todo.

    Args:
        serie (pandas.Series): Índice de fechas. Los NaN se ignoran (se necesitan al menos 3 valores).

    Returns:
        Tendencia: `pendiente_anual`, `intercepto`, `r2`, `p_valor`, `n` y `ajuste`
        (la recta, para dibujarla junto a la serie).
    """
    from scipy import stats

    serie = serie.dropna().sort_index()
    if len(serie) < 3:
        raise ValueError("Se necesitan al menos 3 valores para ajustar una tendencia.")
    anios = (serie.index - serie.index[0]).total_seconds().to_numpy() / (365.25 * 86400)
    ajuste = stats.linregress(anios, serie.to_numpy(dtype=float))
    recta = pd.Series(ajuste.intercept + ajuste.slope * anios, index=serie.index, name="tendencia")
    return Tendencia(ajuste.slope, ajuste.intercept, ajuste.rvalue**2, ajuste.pvalue, len(serie), recta)


def climatologia_mensual(serie, base=None):
    """
    Media, desviación estándar y número de observaciones de cada mes del año (1 a 12).

    Args:
        serie (pandas.Series): Índice de fechas.
        base (tuple, optional): Años (inicio, fin), incluidos, que definen la
            climatología. Por defecto toda la serie.

    Returns:
        pandas.DataFrame: Índice `mes` (1 a 12) y columnas `media`, `desv` y `n`.
        La desviación necesita al menos 2 observaciones del mes (si no, NaN).
    """
    serie = serie.dropna()
    if base is not None:
        serie = serie[(serie.index.year >= base[0]) & (serie.index.year <= base[1])]
    grupo = serie.groupby(serie.index.month)
    tabla = pd.DataFrame({"media": grupo.mean(), "desv": grupo.std(), "n": grupo.size()})
    return tabla.reindex(pd.RangeIndex(1, 13, name="mes"))


def anomalias_zscore(serie, referencia="mes", base=None):
    """
    Anomalías estandarizadas (z-score): cuántas desviaciones se aleja cada valor de lo normal.

    z = (valor - media) / desviación, donde la media y la desviación son las del
    mismo mes del año (`referencia="mes"`, quita el ciclo estacional) o las de toda
    la serie (`referencia="total"`). Un |z| mayor que 2 es poco habitual.

    Con pocos años, cada mes tiene pocas observaciones y su desviación es poco fiable
    (con un solo año, `referencia="mes"` mide solo la variación dentro del mes);
    una climatología mensual seria necesita varios años (idealmente 30).

    Args:
        serie (pandas.Series): Índice de fechas.
        referencia (str): "mes" o "total".
        base (tuple, optional): Años (inicio, fin) con que se calculan la media y la
            desviación. Por defecto toda la serie.

    Returns:
        pandas.Series: z-score, con el mismo índice que `serie` (NaN si la
        desviación de referencia no existe o es cero).
    """
    if referencia not in ("mes", "total"):
        raise ValueError("referencia debe ser 'mes' o 'total'.")
    en_base = serie.dropna()
    if base is not None:
        en_base = en_base[(en_base.index.year >= base[0]) & (en_base.index.year <= base[1])]
    if referencia == "total":
        media, desv = en_base.mean(), en_base.std()
        z = (serie - media) / (desv if desv else np.nan)
    else:
        clima = climatologia_mensual(en_base)
        media = serie.index.month.map(clima["media"])
        desv = serie.index.month.map(clima["desv"]).to_numpy(dtype=float)
        z = (serie - np.asarray(media, dtype=float)) / np.where(desv > 0, desv, np.nan)
    z.name = f"{serie.name}_z" if serie.name else "z"
    return z
