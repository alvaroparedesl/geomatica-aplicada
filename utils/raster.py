"""
Funciones de apoyo para imágenes satelitales (Landsat y Sentinel-2).

Todas trabajan con datos de xarray cargados con `odc.stac.load` (Dataset con
dimensiones time, y, x y una variable por banda). En Sentinel-2 conviene preparar
los datos con `utils.stac.agregar_offset` antes de convertir a reflectancia.
"""

import warnings

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (activa el accesor .rio)
import xarray as xr
from pyproj import Transformer

# Coordenada (una por fecha) con el offset de reflectancia de Sentinel-2.
COORD_OFFSET = "offset_reflectancia"

# ---------------------------------------------------------------------------
# Configuración de cada sensor
# ---------------------------------------------------------------------------
SENSORES = {
    "sentinel-2": {
        "descripcion": "Sentinel-2 L2A (ESA) - Resolución 10 m",
        "coleccion": "sentinel-2-l2a",
        "plataforma_filtro": None,
        # Nombres de las bandas en el catálogo STAC
        "bandas": {
            "red": "B04",
            "green": "B03",
            "blue": "B02",
            "nir": "B08",
            "swir1": "B11",
            "swir2": "B12",
            "red_edge1": "B05",
        },
        "rgb_bandas": ["B04", "B03", "B02"],
        "falso_color": ["B08", "B04", "B03"],  # NIR, rojo, verde
        # reflectancia = DN * factor + offset. En Sentinel-2 el offset depende de
        # cada imagen (0 o -0.1, según el catálogo y el baseline de procesamiento;
        # ver `utils.stac.offset_sentinel2`): `utils.stac.agregar_offset` lo guarda
        # por fecha en la coordenada "offset_reflectancia". El -0.1 de aquí es el
        # valor de las imágenes con baseline 04.00 o posterior (desde el 25-01-2022).
        "factor_reflectancia": 0.0001,
        "offset": -0.1,
        "offset_por_imagen": True,
        "rango_valido": (0, 1),
        # Banda de calidad usada para enmascarar nubes
        "banda_calidad": "SCL",
    },
    "landsat": {
        "descripcion": "Landsat 8/9 Collection 2 L2 (USGS) - Resolución 30 m",
        "coleccion": "landsat-c2-l2",
        "plataforma_filtro": ["landsat-8", "landsat-9"],
        "bandas": {
            "coastal": "coastal",
            "red": "red",
            "green": "green",
            "blue": "blue",
            "nir": "nir08",
            "swir1": "swir16",
            "swir2": "swir22",
        },
        "rgb_bandas": ["red", "green", "blue"],
        "falso_color": ["nir08", "red", "green"],
        "factor_reflectancia": 0.0000275,
        "offset": -0.2,
        "rango_valido": (0, 1),
        "banda_calidad": "qa_pixel",
    },
}


def obtener_configuracion_sensor(sensor):
    """
    Devuelve la configuración de un sensor.

    Args:
        sensor (str o dict): "landsat" o "sentinel-2". Si ya es un diccionario
            de configuración, se devuelve tal cual.

    Returns:
        dict: Bandas, factores de reflectancia, colección STAC, etc.
    """
    if isinstance(sensor, dict):
        return sensor
    if sensor not in SENSORES:
        raise ValueError(f"Sensor '{sensor}' no soportado. Use 'sentinel-2' o 'landsat'.")
    return SENSORES[sensor]


def nombres_bandas(sensor, nombres=("red", "green", "blue", "nir", "swir1"), con_calidad=False):
    """
    Traduce nombres genéricos de bandas a los nombres del sensor.

    Ejemplo: para Landsat, "nir" es "nir08"; para Sentinel-2 es "B08".

    Args:
        sensor (str o dict): "landsat" o "sentinel-2".
        nombres (iterable): Nombres genéricos ("red", "green", "blue", "nir", "swir1", ...).
        con_calidad (bool): Si es True, agrega al final la banda de calidad (nubes).

    Returns:
        list: Nombres de banda listos para pasar a `utils.stac.parametros_carga`.
    """
    config = obtener_configuracion_sensor(sensor)
    bandas = [config["bandas"][nombre] for nombre in nombres]
    if con_calidad:
        bandas.append(config["banda_calidad"])
    return bandas


# ---------------------------------------------------------------------------
# Reflectancia
# ---------------------------------------------------------------------------
def _offset_por_fecha(imagen):
    """
    Offset de Sentinel-2 deducido solo de la fecha (para datos sin la
    coordenada `offset_reflectancia`): -0.1 desde el 25-01-2022, 0 antes.

    Es una aproximación: falla con imágenes antiguas reprocesadas (baseline
    04.00 o posterior, como las de diciembre de 2021 en Planetary Computer).
    """
    if "time" not in imagen.coords:
        raise ValueError(
            "No se puede saber el offset de reflectancia de Sentinel-2: los datos no tienen "
            f"la coordenada '{COORD_OFFSET}' ni fechas. Agrega el offset con "
            "utils.stac.agregar_offset(datos, items) y convierte a reflectancia antes de combinar "
            "fechas (por ejemplo, antes de .median('time')), o indica offset=... explícitamente."
        )
    warnings.warn(
        f"Los datos no tienen la coordenada '{COORD_OFFSET}' (falta llamar a "
        "utils.stac.agregar_offset). Se usa un offset según la fecha (-0.1 desde el "
        "25-01-2022, 0 antes), que puede fallar en imágenes reprocesadas.",
        stacklevel=3,
    )
    fechas = imagen["time"]
    return xr.where(fechas >= np.datetime64("2022-01-25"), -0.1, 0.0)


def convertir_a_reflectancia(imagen, sensor, offset=None):
    """
    Convierte números digitales (DN) a reflectancia: DN * factor + offset.

    Los píxeles sin datos (DN = 0) y los valores fuera del rango válido (0 a 1)
    se convierten en NaN.

    En Sentinel-2 el offset cambia de una imagen a otra (0 o -0.1, ver
    `utils.stac.offset_sentinel2`). Se toma de la coordenada
    `offset_reflectancia` que agrega `utils.stac.agregar_offset` (un valor por
    fecha), así que conviene convertir antes de combinar fechas.

    Args:
        imagen (xarray.Dataset o DataArray): Bandas en DN.
        sensor (str o dict): "landsat" o "sentinel-2".
        offset (float o xarray.DataArray, optional): Offset a usar en vez del
            automático (por ejemplo 0.0 o -0.1).

    Returns:
        Mismo tipo que `imagen`, con reflectancia.
    """
    config = obtener_configuracion_sensor(sensor)
    if offset is None:
        if COORD_OFFSET in imagen.coords:
            offset = imagen.coords[COORD_OFFSET]
        elif config.get("offset_por_imagen"):
            offset = _offset_por_fecha(imagen)
        else:
            offset = config["offset"]
    refl = imagen.where(imagen != 0) * config["factor_reflectancia"] + offset
    minimo, maximo = config["rango_valido"]
    refl = refl.where((refl >= minimo) & (refl <= maximo))
    return refl.drop_vars(COORD_OFFSET, errors="ignore")


# ---------------------------------------------------------------------------
# Máscaras de calidad (nubes, sombras, etc.)
# ---------------------------------------------------------------------------
# Landsat Collection 2: cada bit de QA_PIXEL indica una condición.
BITS_LANDSAT = {
    "relleno": 0,
    "nube_dilatada": 1,
    "cirrus": 2,
    "nube": 3,
    "sombra_nube": 4,
    "nieve": 5,
    "agua": 7,
}
# Sentinel-2 L2A: la banda SCL clasifica cada píxel en una clase.
CLASES_SCL = {
    "sin_datos": 0,
    "saturado": 1,
    "sombra_nube": 3,
    "nube_media": 8,
    "nube_alta": 9,
    "cirrus": 10,
    "nieve": 11,
}
_EXCLUIR_LANDSAT = ("relleno", "nube_dilatada", "cirrus", "nube", "sombra_nube")
_EXCLUIR_SCL = ("sin_datos", "saturado", "sombra_nube", "nube_media", "nube_alta", "cirrus")


def mascara_calidad_landsat(qa_pixel, excluir=None):
    """
    Máscara de píxeles válidos a partir de la banda QA_PIXEL de Landsat.

    Args:
        qa_pixel (xarray.DataArray): Banda "qa_pixel".
        excluir (iterable): Condiciones a descartar. Pueden ser nombres de
            `BITS_LANDSAT` ("nube", "sombra_nube", "nieve", ...) o números de bit.
            Por defecto: relleno, nube dilatada, cirrus, nube y sombra de nube.

    Returns:
        xarray.DataArray (bool): True donde el píxel es válido.
    """
    valida = xr.ones_like(qa_pixel, dtype=bool)
    if excluir is None:
        excluir = _EXCLUIR_LANDSAT
    for condicion in excluir:
        bit = BITS_LANDSAT[condicion] if isinstance(condicion, str) else condicion
        valida = valida & ((qa_pixel.astype("uint16") & (1 << bit)) == 0)
    return valida


def mascara_calidad_sentinel2(scl, excluir=None):
    """
    Máscara de píxeles válidos a partir de la banda SCL de Sentinel-2.

    Args:
        scl (xarray.DataArray): Banda "SCL" (clasificación de escena).
        excluir (iterable): Clases a descartar. Pueden ser nombres de
            `CLASES_SCL` ("nube_alta", "sombra_nube", ...) o códigos numéricos.
            Por defecto: sin datos, saturado, sombra de nube, nubes y cirrus.

    Returns:
        xarray.DataArray (bool): True donde el píxel es válido.
    """
    if excluir is None:
        excluir = _EXCLUIR_SCL
    codigos = [CLASES_SCL[c] if isinstance(c, str) else c for c in excluir]
    return ~scl.isin(codigos)


def enmascarar_nubes(imagen, sensor, excluir=None):
    """
    Descarta (NaN) los píxeles con nubes, sombras y otros problemas.

    Usa la banda de calidad del propio Dataset (qa_pixel o SCL), que debe haberse
    cargado junto con las demás bandas (ver `nombres_bandas(..., con_calidad=True)`).
    La banda de calidad se elimina del resultado.

    Args:
        imagen (xarray.Dataset): Bandas + banda de calidad.
        sensor (str o dict): "landsat" o "sentinel-2".
        excluir (iterable, optional): Condiciones a descartar. Si es None se usan
            las estándar (relleno, nubes, cirrus y sombras).

    Returns:
        xarray.Dataset: Bandas espectrales con NaN en los píxeles descartados.
    """
    config = obtener_configuracion_sensor(sensor)
    banda_calidad = config["banda_calidad"]
    if banda_calidad not in imagen and banda_calidad.lower() in imagen:
        banda_calidad = banda_calidad.lower()  # "scl" en Earth Search
    if banda_calidad not in imagen:
        raise ValueError(f"El Dataset no contiene la banda de calidad '{banda_calidad}'.")

    calidad = imagen[banda_calidad]
    if config["banda_calidad"] == "qa_pixel":
        valida = mascara_calidad_landsat(calidad, excluir)
    else:
        valida = mascara_calidad_sentinel2(calidad, excluir)
    return imagen.drop_vars(banda_calidad).where(valida)


def preparar_reflectancia(imagen, sensor, enmascarar=True, excluir=None):
    """
    Atajo: enmascara nubes (opcional) y convierte a reflectancia.

    Args:
        imagen (xarray.Dataset): Bandas en DN (+ banda de calidad si `enmascarar`).
        sensor (str o dict): "landsat" o "sentinel-2".
        enmascarar (bool): Si es True aplica `enmascarar_nubes`.
        excluir (iterable, optional): Ver `enmascarar_nubes`.

    Returns:
        xarray.Dataset: Reflectancia (0 a 1), con NaN en píxeles no válidos.
    """
    if enmascarar:
        imagen = enmascarar_nubes(imagen, sensor, excluir)
    return convertir_a_reflectancia(imagen, sensor)


# ---------------------------------------------------------------------------
# Índices espectrales
# ---------------------------------------------------------------------------
def _finalizar_indice(valores, nombre, nombre_largo, formula):
    """Descarta valores fuera de [-1, 1] y agrega metadatos."""
    indice = valores.where((valores >= -1) & (valores <= 1))
    indice.name = nombre
    indice.attrs = {"long_name": nombre_largo, "formula": formula, "range": "-1 to +1"}
    return indice


def _diferencia_normalizada(a, b):
    """(a - b) / (a + b), con NaN donde el denominador es cero."""
    suma = a + b
    return (a - b) / suma.where(suma != 0)


def calcular_ndvi(imagen, sensor):
    """
    NDVI = (NIR - ROJO) / (NIR + ROJO)

    Args:
        imagen (xarray.Dataset): Reflectancia (ver `convertir_a_reflectancia`).
        sensor (str o dict): "landsat" o "sentinel-2".

    Returns:
        xarray.DataArray: NDVI entre -1 y 1.
    """
    b = obtener_configuracion_sensor(sensor)["bandas"]
    valores = _diferencia_normalizada(imagen[b["nir"]], imagen[b["red"]])
    return _finalizar_indice(
        valores, "NDVI", "Normalized Difference Vegetation Index", "(NIR - RED) / (NIR + RED)"
    )


def calcular_evi(imagen, sensor):
    """
    EVI = 2.5 * (NIR - ROJO) / (NIR + 6*ROJO - 7.5*AZUL + 1)

    Args:
        imagen (xarray.Dataset): Reflectancia.
        sensor (str o dict): "landsat" o "sentinel-2".

    Returns:
        xarray.DataArray: EVI (se descartan valores fuera de -1 a 1).
    """
    b = obtener_configuracion_sensor(sensor)["bandas"]
    nir, rojo, azul = imagen[b["nir"]], imagen[b["red"]], imagen[b["blue"]]
    denominador = nir + 6 * rojo - 7.5 * azul + 1
    valores = 2.5 * (nir - rojo) / denominador.where(denominador != 0)
    return _finalizar_indice(
        valores,
        "EVI",
        "Enhanced Vegetation Index",
        "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
    )


def calcular_ndwi(imagen, sensor):
    """
    NDWI = (NIR - SWIR1) / (NIR + SWIR1)  (contenido de agua en la vegetación)

    Args:
        imagen (xarray.Dataset): Reflectancia.
        sensor (str o dict): "landsat" o "sentinel-2".

    Returns:
        xarray.DataArray: NDWI entre -1 y 1.
    """
    b = obtener_configuracion_sensor(sensor)["bandas"]
    valores = _diferencia_normalizada(imagen[b["nir"]], imagen[b["swir1"]])
    return _finalizar_indice(
        valores, "NDWI", "Normalized Difference Water Index", "(NIR - SWIR1) / (NIR + SWIR1)"
    )


def calcular_ndwi_agua(imagen, sensor):
    """
    NDWI de McFeeters = (VERDE - NIR) / (VERDE + NIR)  (cuerpos de agua)

    Es el índice de agua abierta: positivo sobre el agua (el verde refleja más que
    el NIR) y negativo sobre tierra y vegetación. No es el `NDWI` de
    `calcular_ndwi` (NIR y SWIR, humedad de la vegetación), que se llama igual.

    Args:
        imagen (xarray.Dataset): Reflectancia.
        sensor (str o dict): "landsat" o "sentinel-2".

    Returns:
        xarray.DataArray: NDWI_AGUA entre -1 y 1.
    """
    b = obtener_configuracion_sensor(sensor)["bandas"]
    valores = _diferencia_normalizada(imagen[b["green"]], imagen[b["nir"]])
    return _finalizar_indice(
        valores, "NDWI_AGUA", "Normalized Difference Water Index (McFeeters)", "(GREEN - NIR) / (GREEN + NIR)"
    )


def calcular_ndci(imagen, sensor):
    """
    NDCI = (BORDE ROJO - ROJO) / (BORDE ROJO + ROJO)  (clorofila-a en agua, Mishra y Mishra 2012)

    Usa la banda B05 (borde rojo, 705 nm) de Sentinel-2, que Landsat no tiene.
    Sube con la clorofila (floraciones de fitoplancton) en aguas costeras.

    Args:
        imagen (xarray.Dataset): Reflectancia con B05 y B04.
        sensor (str o dict): Solo "sentinel-2".

    Returns:
        xarray.DataArray: NDCI entre -1 y 1.
    """
    b = obtener_configuracion_sensor(sensor)["bandas"]
    if "red_edge1" not in b:
        raise ValueError("El NDCI necesita la banda de borde rojo: solo existe en Sentinel-2.")
    valores = _diferencia_normalizada(imagen[b["red_edge1"]], imagen[b["red"]])
    return _finalizar_indice(
        valores, "NDCI", "Normalized Difference Chlorophyll Index", "(RED_EDGE1 - RED) / (RED_EDGE1 + RED)"
    )


_INDICES = {
    "NDVI": calcular_ndvi,
    "EVI": calcular_evi,
    "NDWI": calcular_ndwi,
    "NDWI_AGUA": calcular_ndwi_agua,
    "NDCI": calcular_ndci,
}


def calcular_indices(imagen, sensor, indices=("NDVI", "EVI", "NDWI")):
    """
    Calcula varios índices y los entrega juntos en un Dataset.

    Args:
        imagen (xarray.Dataset): Reflectancia.
        sensor (str o dict): "landsat" o "sentinel-2".
        indices (iterable): Cualquiera de "NDVI", "EVI", "NDWI", "NDWI_AGUA" o "NDCI".

    Returns:
        xarray.Dataset: Una variable por índice.
    """
    return xr.Dataset({nombre: _INDICES[nombre](imagen, sensor) for nombre in indices})


# ---------------------------------------------------------------------------
# Coordenadas, puntos y series temporales
# ---------------------------------------------------------------------------
def lonlat_a_crs(lon, lat, crs_destino):
    """
    Transforma coordenadas geográficas (EPSG:4326) a otro CRS.

    Args:
        lon, lat (float o lista): Longitud y latitud en grados.
        crs_destino (str): Por ejemplo "EPSG:32719" (UTM 19S).

    Returns:
        tuple: (x, y) en el CRS de destino.
    """
    transformador = Transformer.from_crs("EPSG:4326", crs_destino, always_xy=True)
    return transformador.transform(lon, lat)


def extraer_series_puntos(datos, puntos, columna_id="punto_id", radio=None, estadistico="mediana", crs=None):
    """
    Extrae la serie temporal de cada punto: el píxel más cercano o un estadístico de una ventana.

    Los puntos pueden venir en coordenadas proyectadas (`x`, `y`, en el CRS de
    `datos`) o geográficas (`lon`, `lat`, en grados), que se transforman al CRS de
    `datos`.

    Args:
        datos (xarray.Dataset): Variables con dimensiones (time, y, x),
            por ejemplo el resultado de `calcular_indices`.
        puntos (pandas.DataFrame): Una fila por punto, con las columnas `x` e `y`
            (en el CRS de `datos`) o `lon` y `lat` (grados), y una columna
            identificadora única. Las demás columnas (tipo de uso, nombre...) se
            conservan en el resultado.
        columna_id (str): Nombre de la columna identificadora.
        radio (float, optional): Radio de la ventana, en unidades del CRS (metros en
            UTM). Si es None (por defecto) se toma el píxel más cercano al punto. Si
            se indica, se resume con `estadistico` los píxeles cuyo centro está a
            `radio` o menos del punto (ignorando NaN, por ejemplo los píxeles
            enmascarados por nubes). Si ningún píxel cae dentro del radio, se usa el
            más cercano.
        estadistico (str): "mediana", "media", "min" o "max" (solo con `radio`).
        crs (str, optional): CRS de `datos` para transformar `lon` y `lat`. Si es None
            se lee de `datos.rio.crs`.

    Returns:
        pandas.DataFrame: Formato largo, con columnas [columna_id, time, x, y,
        indice, valor] más las columnas extra de `puntos`. Con `radio`, `x` e `y`
        son las del punto y se agrega `n_validos` (píxeles con dato en la ventana).
    """
    ids = puntos[columna_id].values
    puntos = puntos.copy()
    if "x" not in puntos or "y" not in puntos:
        crs = crs or datos.rio.crs
        puntos["x"], puntos["y"] = lonlat_a_crs(puntos["lon"].values, puntos["lat"].values, crs)

    if radio is not None:
        return _extraer_ventanas(datos, puntos, columna_id, radio, estadistico)

    seleccion = datos.sel(
        x=xr.DataArray(puntos["x"].values, dims=columna_id, coords={columna_id: ids}),
        y=xr.DataArray(puntos["y"].values, dims=columna_id, coords={columna_id: ids}),
        method="nearest",
    )
    serie = (
        seleccion.to_dataframe()
        .reset_index()
        .melt(
            id_vars=[columna_id, "time", "x", "y"],
            value_vars=list(datos.data_vars),
            var_name="indice",
            value_name="valor",
        )
    )
    extras = puntos.drop(columns=["x", "y"])
    return serie.merge(extras, on=columna_id, how="left")


_ESTADISTICOS = {"mediana": "median", "media": "mean", "min": "min", "max": "max"}


def _extraer_ventanas(datos, puntos, columna_id, radio, estadistico):
    """Resume una ventana circular de píxeles alrededor de cada punto (ver `extraer_series_puntos`)."""
    if estadistico not in _ESTADISTICOS:
        raise ValueError(f"estadistico debe ser uno de {list(_ESTADISTICOS)}, no {estadistico!r}.")
    variables = list(datos.data_vars)
    tablas = []
    for identificador, x, y in zip(puntos[columna_id], puntos["x"], puntos["y"]):
        distancia = np.hypot(datos["x"] - x, datos["y"] - y)
        dentro = distancia <= radio
        if not bool(dentro.any()):  # radio menor que un píxel: el más cercano
            dentro = distancia == distancia.min()
        ventana = datos[variables].where(dentro)
        if ventana[variables[0]].chunks is not None:  # dask: la mediana necesita la ventana en un solo bloque
            ventana = ventana.chunk({"y": -1, "x": -1})
        resumen = getattr(ventana, _ESTADISTICOS[estadistico])(dim=("y", "x"), skipna=True)
        cuenta = ventana.count(dim=("y", "x"))
        valores = resumen[variables].to_dataframe().reset_index().melt(
            id_vars="time", value_vars=variables, var_name="indice", value_name="valor")
        validos = cuenta[variables].to_dataframe().reset_index().melt(
            id_vars="time", value_vars=variables, var_name="indice", value_name="n_validos")
        tabla = valores.merge(validos, on=["time", "indice"])
        tabla.insert(0, columna_id, identificador)
        tabla["x"], tabla["y"] = x, y
        tablas.append(tabla)
    serie = pd.concat(tablas, ignore_index=True)
    serie = serie[[columna_id, "time", "x", "y", "indice", "valor", "n_validos"]]
    return serie.merge(puntos.drop(columns=["x", "y"]), on=columna_id, how="left")


def recortar_region(imagen, limites, crs=None):
    """
    Recorta una imagen usando un rectángulo en coordenadas geográficas.

    Args:
        imagen (xarray.Dataset o DataArray): Imagen con ejes x, y (y decreciente).
        limites (dict): Con claves lon_min, lon_max, lat_min, lat_max (grados),
            por ejemplo `selector.limites` de `utils.viz.SelectorRegion`.
        crs (str, optional): CRS de la imagen. Si es None se lee de la imagen.

    Returns:
        Imagen recortada.
    """
    crs = crs or imagen.rio.crs
    x1, y1 = lonlat_a_crs(limites["lon_min"], limites["lat_min"], crs)
    x2, y2 = lonlat_a_crs(limites["lon_max"], limites["lat_max"], crs)
    return imagen.sel(x=slice(min(x1, x2), max(x1, x2)), y=slice(max(y1, y2), min(y1, y2)))


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------
def calcular_estadisticas(dataset, bandas=None):
    """
    Estadísticas descriptivas de cada banda (ignora NaN).

    Args:
        dataset (xarray.Dataset): Bandas en reflectancia (sin dimensión time).
        bandas (list, optional): Bandas a analizar. Por defecto todas.

    Returns:
        pandas.DataFrame: Una fila por banda: media, mediana, desviación,
        percentiles, asimetría, curtosis, etc.
    """
    from scipy import stats

    filas = []
    for banda in bandas or list(dataset.data_vars):
        datos = dataset[banda].values.ravel()
        datos = datos[~np.isnan(datos)]
        if len(datos) == 0:
            continue
        q1, q3 = np.percentile(datos, [25, 75])
        filas.append(
            {
                "Banda": banda,
                "N_pixels": len(datos),
                "Media": datos.mean(),
                "Mediana": np.median(datos),
                "Desv_std": datos.std(),
                "Varianza": datos.var(),
                "Mínimo": datos.min(),
                "Máximo": datos.max(),
                "Rango": datos.max() - datos.min(),
                "Q1": q1,
                "Q3": q3,
                "IQR": q3 - q1,
                "Asimetría": stats.skew(datos),
                "Curtosis": stats.kurtosis(datos),
                "CV_%": datos.std() / datos.mean() * 100,
            }
        )
    return pd.DataFrame(filas)
