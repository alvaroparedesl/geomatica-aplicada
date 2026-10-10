"""
Apoyo para buscar y cargar imágenes satelitales desde catálogos STAC.

Estas funciones NO reemplazan las llamadas nativas: las preparan. Las llamadas
a `pystac_client` y `odc.stac` las escribes tú, y los helpers solo arman sus
argumentos y ordenan los datos antes y después:

    catalogo = conectar_catalogo()                       # pystac_client.Client
    busqueda = catalogo.search(**parametros_busqueda("sentinel-2", bbox, (inicio, fin),
                                                     nubes_max=20, catalogo=catalogo))
    items = busqueda.item_collection()                   # pystac.ItemCollection
    items = preparar_items(items)                        # sin duplicados ni catálogos mezclados
    datos = odc.stac.load(items, **parametros_carga("sentinel-2", ["B04", "B08", "SCL"],
                                                    resolucion=30, crs="EPSG:32719", bbox=bbox))
    datos = renombrar_bandas(datos)                      # B04, B08, SCL en todos los catálogos
    datos = agregar_offset(datos, items)                 # offset de reflectancia de cada fecha
    refl = utils.raster.convertir_a_reflectancia(datos, "sentinel-2")

Catálogos de Sentinel-2 L2A (`CATALOGO_SENTINEL2` elige el que se usa por defecto):

* "planetary": Microsoft Planetary Computer (colección `sentinel-2-l2a`).
* "earthsearch": Element84 Earth Search en AWS, Colección 1 (`sentinel-2-c1-l2a`).
* "earthsearch-l2a": Earth Search `sentinel-2-l2a`. DESACTIVADO: sus metadatos de
  offset no son confiables (ver más abajo). Elegirlo da un ValueError.

Landsat y el DEM Copernicus se buscan siempre en Planetary Computer.

===========================================================================
OFFSET DE REFLECTANCIA DE SENTINEL-2 (BOA_ADD_OFFSET)
===========================================================================
Desde el baseline de procesamiento 04.00 (introducido el 25-01-2022) la ESA suma
1000 a los números digitales (DN) de las bandas L2A, así que la reflectancia es
`DN * 0.0001 - 0.1`. Con baseline anterior es `DN * 0.0001`. Aplicar mal el
offset desplaza toda la reflectancia en 0.1: el agua oscura queda negativa (NaN)
o los índices cambian de una fecha a otra. Cada catálogo entrega los píxeles de
forma distinta, y la regla de `offset_sentinel2` por catálogo es:

| Catálogo (colección)             | Regla                                                    |
|----------------------------------|----------------------------------------------------------|
| planetary (sentinel-2-l2a)       | -0.1 si `s2:processing_baseline` >= 04.00, si no 0       |
| earthsearch (sentinel-2-c1-l2a)  | siempre -0.1 (todo es baseline >= 05.00)                 |
| earthsearch-l2a (desactivado)    | casi siempre 0; -0.1 solo en baseline 04.00 entre el     |
|                                  | 25-01-2022 y fines de feb-2022 si el indicador            |
|                                  | `earthsearch:boa_offset_applied` no es True               |

Todas las reglas se comprobaron sobre píxeles reales (mar abierto y tierra en
Concepción, y valle del Maipo contra Landsat). Lo que hay detrás de cada una:

1. Planetary Computer. Los metadatos no traen `raster:bands` con offset, y los
   píxeles llevan el +1000 en todo baseline >= 04.00, incluidas las tomas de
   diciembre de 2021 reprocesadas más tarde con 04.00 (por eso decide el
   baseline y NO la fecha de la toma).
   Pruebas: `test_offset_sentinel2.py::test_tabla_de_offsets` (filas "Planetary
   Computer"), `test_todos_los_items_tienen_offset_valido`,
   `test_pixeles_reales.py` (la reflectancia coincide con la de los otros
   catálogos) y `test_pixeles_reales.py::test_un_offset_equivocado_se_detecta`.

2. Earth Search Colección 1 (`earthsearch`). Todo el catálogo está reprocesado
   con baseline >= 05.00 y los píxeles siempre llevan el +1000, así que el
   offset es siempre -0.1. Además esta colección tiene un HUECO: entre el
   29-12-2021 y el 07-12-2022 (`HUECO_C1`; esas dos fechas sí tienen tomas) casi
   no hay imágenes, solo unas pocas escenas sueltas (por ejemplo, de marzo de
   2022). `parametros_busqueda` avisa si el período pedido toca el hueco; para
   ese período usa "planetary".
   Pruebas: filas "c1_*" de `test_tabla_de_offsets`, `test_pixeles_reales.py`
   (c1 contra Planetary Computer) y `test_busqueda_offline.py::test_aviso_hueco_c1`.

3. Earth Search `sentinel-2-l2a` (desactivado). Sus metadatos NO sirven para decidir
   el offset: `earthsearch:boa_offset_applied` dice False en muchas imágenes cuyos
   píxeles ya vienen corregidos (en la escena de Concepción, 33 de 45 ítems dicen
   False y están bien), y el `offset` de `raster:bands` dice -0.1 aunque el desfase
   ya se haya restado. Midiendo píxeles se vio que Earth Search dejó el +1000 solo
   en tomas de baseline 04.00 entre el 25-01-2022 y fines de febrero de 2022
   (las últimas comprobadas son del 25 y 26 de febrero); después restó el desfase
   aunque el indicador diga False. Como la única señal confiable es el indicador
   en True (y la ventana de fechas), se desactivó para el curso: el conocimiento
   queda en `offset_sentinel2` y en sus pruebas, pero no se puede elegir.
   Pruebas: `test_metadatos_earthsearch_no_son_confiables`,
   `test_limites_de_la_ventana_earthsearch`, `test_earthsearch_sin_propiedad_aplicada`,
   filas "es_*" de `test_tabla_de_offsets` y `test_pixeles_reales.py`.

Cómo se aplica por fecha: `parametros_carga` entrega un `groupby` que agrupa por
(día solar, offset), así `odc.stac.load` nunca mezcla en una misma fecha imágenes
con offsets distintos; `agregar_offset` guarda el offset de cada fecha en la
coordenada `offset_reflectancia` y `utils.raster.convertir_a_reflectancia` lo usa.
Pruebas: `test_carga_offline.py` (ver también `utils/README.md`, sección "Pruebas").
"""

import datetime as dt
import re
import warnings
from urllib.parse import urlparse

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
from pystac_client import Client

from .raster import COORD_OFFSET, obtener_configuracion_sensor

# ---------------------------------------------------------------------------
# Catálogos disponibles
# ---------------------------------------------------------------------------
URL_PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
URL_EARTHSEARCH = "https://earth-search.aws.element84.com/v1"

CATALOGOS = {
    "planetary": {
        "descripcion": "Microsoft Planetary Computer (enlaces firmados, sin cuenta)",
        "url": URL_PLANETARY,
        "colecciones": {"sentinel-2": "sentinel-2-l2a", "landsat": "landsat-c2-l2"},
    },
    "earthsearch": {
        "descripcion": "Element84 Earth Search en AWS, Colección 1 (sentinel-2-c1-l2a, sin firma ni cuenta)",
        "url": URL_EARTHSEARCH,
        "colecciones": {"sentinel-2": "sentinel-2-c1-l2a"},
    },
    "earthsearch-l2a": {
        "descripcion": "Element84 Earth Search en AWS, sentinel-2-l2a (DESACTIVADO)",
        "url": URL_EARTHSEARCH,
        "colecciones": {"sentinel-2": "sentinel-2-l2a"},
        # Si hay un texto aquí, el catálogo no se puede elegir (ver `_nombre_catalogo`).
        "desactivado": (
            "El catálogo 'earthsearch-l2a' (sentinel-2-l2a de Earth Search) está desactivado: "
            "sus metadatos de offset de reflectancia son inconsistentes (el indicador "
            "'boa_offset_applied' y el offset de 'raster:bands' no coinciden con los píxeles). "
            "Usa 'earthsearch' (Colección 1) o 'planetary'."
        ),
    },
}

# Catálogo usado para Sentinel-2 cuando no se indica otro. Para cambiarlo en
# todo el curso, edita solo esta línea ("planetary" o "earthsearch").
# (Landsat, el DEM y las demás colecciones siguen en Planetary Computer.)
CATALOGO_SENTINEL2 = "planetary"

# Nombres de los archivos (assets) de Earth Search para cada banda Sentinel-2.
# Las bandas se entregan con los nombres de la izquierda (los de la ESA).
BANDAS_EARTHSEARCH = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
    "AOT": "aot",
    "WVP": "wvp",
    "visual": "visual",
}

# Offset (en reflectancia) de las imágenes con baseline 04.00 o posterior.
OFFSET_BASELINE_04 = -0.1
# La ESA empezó a usar el baseline 04.00 el 25-01-2022.
INICIO_BASELINE_04 = dt.date(2022, 1, 25)
# Earth Search sentinel-2-l2a dejó el +1000 (sin restar) solo en tomas hasta
# esta fecha; ver `offset_sentinel2`.
FIN_SIN_RESTAR_EARTHSEARCH = dt.date(2022, 2, 28)
# Earth Search sentinel-2-c1-l2a casi no tiene imágenes entre estas dos fechas (las dos
# fechas mismas sí tienen tomas). Hay solo unas pocas escenas sueltas dentro del hueco.
HUECO_C1 = (dt.date(2021, 12, 29), dt.date(2022, 12, 7))


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------
def _nombre_catalogo(catalogo):
    """
    Clave de `CATALOGOS` para buscar Sentinel-2.

    None significa "el catálogo por defecto" (`CATALOGO_SENTINEL2`). Un cliente
    STAC se reconoce por su URL. Un catálogo desactivado da ValueError.
    """
    if catalogo is None:
        catalogo = CATALOGO_SENTINEL2
    if not isinstance(catalogo, str):
        href = catalogo.get_self_href() or ""
        catalogo = "earthsearch" if "earth-search" in href else "planetary"
    if catalogo not in CATALOGOS:
        validos = ", ".join(n for n, c in CATALOGOS.items() if not c.get("desactivado"))
        raise ValueError(f"Catálogo '{catalogo}' no reconocido. Usa uno de: {validos}.")
    if CATALOGOS[catalogo].get("desactivado"):
        raise ValueError(CATALOGOS[catalogo]["desactivado"])
    return catalogo


def conectar_catalogo(catalogo=None):
    """
    Conecta a un catálogo STAC y devuelve el cliente nativo de `pystac_client`.

    Equivale a `Client.open(url, ...)`; en Planetary Computer agrega
    `modifier=planetary_computer.sign_inplace` para firmar los enlaces.

    Args:
        catalogo (str, optional): "planetary" o "earthsearch". Si es None se usa
            `CATALOGO_SENTINEL2`. Para Landsat, el DEM, ESA WorldCover, etc.
            indica "planetary" explícitamente.

    Returns:
        pystac_client.Client: Catálogo listo para hacer búsquedas.
    """
    url = CATALOGOS[_nombre_catalogo(catalogo)]["url"]
    if url == URL_PLANETARY:
        return Client.open(url, modifier=planetary_computer.sign_inplace)
    return Client.open(url)


# ---------------------------------------------------------------------------
# Metadatos de Sentinel-2: catálogo, baseline, offset y duplicados
# ---------------------------------------------------------------------------
def es_sentinel2(item):
    """True si el ítem STAC es una imagen Sentinel-2."""
    return (item.collection_id or "").startswith("sentinel-2")


def catalogo_de_item(item):
    """
    Identifica de qué catálogo viene un ítem Sentinel-2.

    Returns:
        str: "planetary", "earthsearch" (Colección 1) o "earthsearch-l2a"
        (sentinel-2-l2a de Earth Search, desactivado pero reconocible).
    """
    if item.collection_id == "sentinel-2-c1-l2a":
        return "earthsearch"
    enlaces = [link.href for link in item.links if link.href] + [a.href for a in item.assets.values()]
    for href in enlaces:
        host = urlparse(href).netloc
        if "earth-search" in host or "amazonaws.com" in host or href.startswith("s3://"):
            return "earthsearch-l2a"
        if "planetarycomputer" in host or host.endswith("blob.core.windows.net"):
            return "planetary"
    if "earthsearch:boa_offset_applied" in item.properties or "earthsearch:s3_path" in item.properties:
        return "earthsearch-l2a"
    raise ValueError(f"No se pudo identificar el catálogo del ítem {item.id}.")


def _version(baseline):
    """'04.00' -> (4, 0). None si no se puede interpretar."""
    try:
        return tuple(int(parte) for parte in str(baseline).split("."))
    except (TypeError, ValueError):
        return None


def baseline_procesamiento(item):
    """Versión de procesamiento de la ESA (`s2:processing_baseline`), ej. '05.11'."""
    return item.properties.get("s2:processing_baseline")


def offset_sentinel2(item):
    """
    Offset de reflectancia correcto para un ítem Sentinel-2 L2A.

    La reflectancia es `DN * 0.0001 + offset`. Las reglas por catálogo, por qué
    los metadatos de Earth Search l2a no sirven y qué pruebas respaldan cada
    regla están explicadas en la sección "OFFSET DE REFLECTANCIA" al inicio de
    este módulo (`help(utils.stac)`). En resumen:

    | Catálogo / colección        | Baseline < 04.00 | Baseline >= 04.00                          |
    |-----------------------------|------------------|--------------------------------------------|
    | planetary, sentinel-2-l2a   | 0                | -0.1                                       |
    | earthsearch, c1-l2a         | (no existe)      | -0.1 (todo está reprocesado >= 05.00)      |
    | earthsearch-l2a (desactiv.) | 0                | 0 (ya restó los 1000), salvo tomas del     |
    |                             |                  | 25-01-2022 al 28-02-2022 con               |
    |                             |                  | `earthsearch:boa_offset_applied` distinto  |
    |                             |                  | de True: -0.1                              |

    Args:
        item (pystac.Item): Imagen Sentinel-2.

    Returns:
        float: 0.0 o -0.1.
    """
    catalogo = catalogo_de_item(item)
    if catalogo == "earthsearch":
        # Regla c1: los píxeles siempre llevan el +1000 (todo es baseline >= 05.00).
        return OFFSET_BASELINE_04
    version = _version(baseline_procesamiento(item))
    if version is None:
        # Sin baseline: se decide por la fecha de adquisición (regla de la ESA).
        fecha = item.datetime.date() if item.datetime else None
        nuevo = fecha is not None and fecha >= INICIO_BASELINE_04
        warnings.warn(
            f"El ítem {item.id} no informa s2:processing_baseline; se supone "
            f"offset {'-0.1' if nuevo else '0'} según su fecha.",
            stacklevel=2,
        )
    else:
        # Regla Planetary Computer (y base de la de Earth Search l2a): decide el
        # baseline, no la fecha de la toma (hay tomas de 2021 reprocesadas con 04.00).
        nuevo = version >= (4, 0)
    if not nuevo:
        return 0.0
    if catalogo == "earthsearch-l2a":
        # Regla Earth Search l2a: el +1000 solo sigue en tomas de la ventana de
        # 2022 y solo si el indicador no es True. El indicador en False NO prueba
        # nada (muchas imágenes ya restadas dicen False): por eso decide la fecha.
        if item.properties.get("earthsearch:boa_offset_applied") is True:
            return 0.0
        fecha = item.datetime.date()
        if not (INICIO_BASELINE_04 <= fecha <= FIN_SIN_RESTAR_EARTHSEARCH):
            return 0.0
    return OFFSET_BASELINE_04


def _inicio_datastrip(item):
    """Instante de sensado del datastrip (igual en todas las versiones de una toma)."""
    datastrip = item.properties.get("s2:datastrip_id") or ""
    coincidencia = re.search(r"_S(\d{8}T\d{6})_", datastrip)
    return coincidencia.group(1) if coincidencia else item.datetime.strftime("%Y%m%d")


def _tile(item):
    p = item.properties
    if p.get("s2:mgrs_tile"):
        return p["s2:mgrs_tile"]
    if p.get("mgrs:grid_square"):
        return f"{p.get('mgrs:utm_zone')}{p.get('mgrs:latitude_band')}{p['mgrs:grid_square']}"
    return p.get("grid:code") or item.id


def quitar_duplicados(items, verbose=False):
    """
    Deja una sola versión de cada toma Sentinel-2.

    La misma toma (satélite + tile + datastrip) puede aparecer varias veces en
    un catálogo: la versión original y los reprocesamientos de la ESA (por
    ejemplo baseline 03.00 y 04.00 en diciembre de 2021, o 04.00 y 05.10 en
    2022-2023). Se conserva la de baseline más alto y, si empatan, la generada
    más tarde. Imágenes de otros satélites u otros datastrips del mismo día se
    conservan. Los ítems que no son Sentinel-2 se devuelven sin cambios.

    Args:
        items (list): Ítems STAC.
        verbose (bool): Si es True, informa cuántos duplicados se quitaron.

    Returns:
        list: Ítems sin duplicados, en el mismo orden original.
    """
    items = list(items)
    mejor = {}
    for posicion, item in enumerate(items):
        if not es_sentinel2(item):
            mejor[("otro", posicion)] = (None, posicion)
            continue
        clave = (
            catalogo_de_item(item),
            (item.properties.get("platform") or "").lower(),
            _tile(item),
            _inicio_datastrip(item),
        )
        rango = (
            _version(baseline_procesamiento(item)) or (0,),
            item.properties.get("s2:generation_time") or "",
            item.id,
        )
        if clave not in mejor or rango > mejor[clave][0]:
            mejor[clave] = (rango, posicion)
    conservar = sorted(posicion for _, posicion in mejor.values())
    if verbose and len(conservar) < len(items):
        print(f"Se quitaron {len(items) - len(conservar)} versiones duplicadas (reprocesamientos).")
    return [items[i] for i in conservar]


# ---------------------------------------------------------------------------
# Búsqueda: parámetros para `catalogo.search(...)` y limpieza de los resultados
# ---------------------------------------------------------------------------
def _separar_fechas(fechas):
    """('2024-01-01', '2024-02-28') o '2024-01-01/2024-02-28' -> (inicio, fin) en texto."""
    if isinstance(fechas, str):
        partes = fechas.split("/")
    else:
        partes = list(fechas)
    if len(partes) != 2:
        raise ValueError("`fechas` debe ser (inicio, fin) o un texto 'AAAA-MM-DD/AAAA-MM-DD'.")
    return str(partes[0]), str(partes[1])


def _avisar_hueco_c1(inicio, fin):
    """Avisa si el período pedido toca el hueco de Earth Search c1 (casi sin imágenes)."""
    try:
        i = pd.Timestamp(inicio).date()
        f = pd.Timestamp(fin).date()
    except (ValueError, TypeError):
        return  # fechas abiertas ("..") u otros formatos: no se revisa
    if i < HUECO_C1[1] and f > HUECO_C1[0]:
        warnings.warn(
            f"Earth Search (Colección 1) casi no tiene imágenes entre {HUECO_C1[0]} y {HUECO_C1[1]}; "
            "para ese período usa el catálogo 'planetary'.",
            stacklevel=3,
        )


def parametros_busqueda(sensor, bbox, fechas, nubes_max=20, catalogo=None):
    """
    Argumentos para `catalogo.search(**parametros)` (no ejecuta la búsqueda).

    Args:
        sensor (str o dict): "landsat" (8 y 9) o "sentinel-2".
        bbox (list): [oeste, sur, este, norte] en grados (EPSG:4326).
        fechas (tuple o str): (inicio, fin) como "AAAA-MM-DD", o "inicio/fin".
        nubes_max (float): Máxima cobertura de nubes de la escena (%).
        catalogo (str o pystac_client.Client, optional): Catálogo de Sentinel-2:
            "planetary", "earthsearch" o el cliente de `conectar_catalogo`. Si es
            None se usa `CATALOGO_SENTINEL2`. Landsat existe solo en Planetary
            Computer (con un cliente de otro catálogo da ValueError).

    Returns:
        dict: `collections`, `bbox`, `datetime` y `query` (el filtro de nubes y,
        en Landsat, de plataforma). Se usa `query` y no `filter` (CQL2) porque
        Earth Search ignora `filter`.
    """
    config = obtener_configuracion_sensor(sensor)
    inicio, fin = _separar_fechas(fechas)
    if config["coleccion"].startswith("sentinel-2"):
        nombre = _nombre_catalogo(catalogo)
        coleccion = CATALOGOS[nombre]["colecciones"]["sentinel-2"]
        if nombre == "earthsearch":
            _avisar_hueco_c1(inicio, fin)
    else:
        if catalogo is not None and _nombre_catalogo(catalogo) != "planetary":
            raise ValueError(
                "Landsat solo está en Planetary Computer: usa conectar_catalogo('planetary')."
            )
        coleccion = config["coleccion"]

    filtros = {"eo:cloud_cover": {"lt": nubes_max}}
    if config["plataforma_filtro"]:
        filtros["platform"] = {"in": config["plataforma_filtro"]}
    return {
        "collections": [coleccion],
        "bbox": bbox,
        "datetime": f"{inicio}/{fin}",
        "query": filtros,
    }


def _catalogo_de_items(items):
    """
    Catálogo único de una lista de ítems Sentinel-2.

    Raises:
        ValueError: Si se mezclan catálogos o el catálogo está desactivado.
    """
    catalogos = {catalogo_de_item(item) for item in items}
    if len(catalogos) > 1:
        raise ValueError(f"No se pueden mezclar imágenes de varios catálogos en una carga: {sorted(catalogos)}.")
    (nombre,) = catalogos
    _nombre_catalogo(nombre)  # lanza ValueError si está desactivado
    return nombre


def preparar_items(items, verbose=True):
    """
    Limpia los ítems encontrados antes de cargarlos con `odc.stac.load`.

    * Sentinel-2: quita versiones duplicadas de una misma toma (`quitar_duplicados`)
      y exige que todos vengan de un mismo catálogo activo (si no, ValueError).
    * Ordena por fecha.

    Args:
        items (iterable): Resultado de `busqueda.item_collection()`.
        verbose (bool): Si es True, informa cuántas imágenes quedaron.

    Returns:
        list: Ítems listos para `odc.stac.load` (puede estar vacía).
    """
    items = list(items)
    sentinel = [item for item in items if es_sentinel2(item)]
    nombre = _catalogo_de_items(sentinel) if sentinel else "planetary"
    total = len(items)
    items = quitar_duplicados(items)
    items.sort(key=lambda item: (item.datetime, item.id))
    if verbose:
        repetidas = f" ({total - len(items)} versiones repetidas descartadas)" if total > len(items) else ""
        colecciones = ", ".join(sorted({item.collection_id for item in items}))
        print(f"Se encontraron {len(items)} imágenes ({colecciones}, {nombre}){repetidas}.")
        if not items:
            print("Prueba ampliando las fechas, subiendo nubes_max o cambiando el área.")
    return items


def resumir_imagenes(items):
    """
    Resume los ítems encontrados en una tabla (fecha, nubes, id).

    Para Sentinel-2 agrega el baseline de procesamiento y el offset de
    reflectancia que corresponde a cada imagen.

    Args:
        items (iterable): Ítems STAC (por ejemplo, el resultado de `preparar_items`).

    Returns:
        pandas.DataFrame: Una fila por imagen, ordenada por fecha.
    """
    items = list(items)
    tabla = pd.DataFrame(
        {
            "fecha": [item.datetime.strftime("%Y-%m-%d") for item in items],
            "nubes_%": [item.properties.get("eo:cloud_cover") for item in items],
            "id": [item.id for item in items],
        }
    )
    if items and all(es_sentinel2(item) for item in items):
        tabla["baseline"] = [baseline_procesamiento(item) for item in items]
        tabla["offset"] = [offset_sentinel2(item) for item in items]
    return tabla.sort_values("fecha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga: parámetros para `odc.stac.load(...)` y arreglo de los datos cargados
# ---------------------------------------------------------------------------
_RIO_CONFIGURADO = False


def configurar_lectura():
    """
    Opciones de GDAL para leer COG por internet (reintentos, sin listar carpetas)
    y lectura anónima de los buckets públicos de AWS (Earth Search).

    `parametros_carga` la llama sola; es segura de llamar varias veces.
    """
    global _RIO_CONFIGURADO
    if not _RIO_CONFIGURADO:
        try:  # con boto3 instalado, rasterio abre una sesión de AWS: que sea anónima
            import boto3  # noqa: F401

            aws = {"aws_unsigned": True}
        except ImportError:
            aws = None
        odc.stac.configure_rio(
            cloud_defaults=True,  # reintentos y no listar carpetas remotas
            aws=aws,
            AWS_NO_SIGN_REQUEST="YES",  # buckets públicos: sin credenciales de AWS
            GDAL_HTTP_CONNECTTIMEOUT=30,
            # cortar (y reintentar) una descarga que se queda "pegada": menos de
            # 1 kB/s durante 60 s. Sin esto una conexión colgada espera para siempre.
            GDAL_HTTP_LOW_SPEED_TIME=60,
            GDAL_HTTP_LOW_SPEED_LIMIT=1024,
        )
        _RIO_CONFIGURADO = True


def firmar_url(href):
    """
    Firma un enlace de Planetary Computer con un token nuevo (para `patch_url`).

    Si el enlace ya traía un token (por ejemplo el que agrega `sign_inplace` al
    buscar) se descarta antes: `planetary_computer.sign` no renueva tokens ya
    presentes, aunque estén vencidos.
    """
    if ".blob.core.windows.net" in href:
        href = href.split("?")[0]
    return planetary_computer.sign(href)


def _dia_solar(item):
    """Fecha local aproximada de la toma (hora UTC corregida por la longitud del ítem)."""
    if item.bbox:
        lon = (item.bbox[0] + item.bbox[2]) / 2
    else:  # ítem sin bbox: centro de su geometría
        from shapely.geometry import shape

        lon = shape(item.geometry).centroid.x
    return (item.datetime + dt.timedelta(hours=lon / 15)).date()


def agrupar_por_dia_y_offset(item, parsed=None, idx=None):
    """
    Clave de agrupación (día solar, offset de reflectancia) de un ítem Sentinel-2.

    Se pasa a `odc.stac.load(..., groupby=agrupar_por_dia_y_offset)` (`parametros_carga`
    lo hace por ti). odc.stac combina en una misma fecha los ítems con la misma
    clave, así que imágenes de un mismo día con offsets distintos (por ejemplo una
    con baseline 03.00 y otra con 04.00) quedan en fechas separadas y nunca se
    mezclan. Los argumentos `parsed` e `idx` los entrega odc.stac y no se usan.
    """
    return (_dia_solar(item).isoformat(), offset_sentinel2(item))


def parametros_carga(sensor, bandas, resolucion, crs, catalogo=None, bbox=None, chunks=None):
    """
    Argumentos para `odc.stac.load(items, **parametros)` (no carga nada).

    Entrega:

    * `bands`: nombres de banda del catálogo (en Earth Search, los nombres de sus
      archivos: "red" en vez de "B04"; `renombrar_bandas` los deja como B04);
    * `crs`, `resolution` y, si se indican, `bbox` y `chunks`;
    * `groupby`: en Sentinel-2 `agrupar_por_dia_y_offset` (nunca mezcla imágenes con
      offsets distintos en una fecha); en Landsat "solar_day";
    * `patch_url`: en Planetary Computer, firma los enlaces con un token nuevo.

    También configura GDAL para leer por internet (`configurar_lectura`).

    Args:
        sensor (str o dict): "landsat" o "sentinel-2".
        bandas (list): Nombres de banda del sensor (ver `utils.raster.nombres_bandas`),
            por ejemplo ["B04", "B08", "SCL"] o ["red", "nir08", "qa_pixel"].
        resolucion (int): Tamaño de píxel en metros.
        crs (str): CRS de salida, por ejemplo "EPSG:32719" (UTM 19S, Chile central).
        catalogo (str o pystac_client.Client, optional): Igual que en
            `parametros_busqueda`. Landsat usa siempre Planetary Computer.
        bbox (list, optional): [oeste, sur, este, norte] en grados. Si se indica,
            se incluye en los parámetros.
        chunks (dict, optional): Por ejemplo {"x": 1024, "y": 1024} para carga
            perezosa con dask. Si es None se carga todo en memoria.

    Returns:
        dict: Argumentos para `odc.stac.load`.
    """
    config = obtener_configuracion_sensor(sensor)
    es_s2 = config["coleccion"].startswith("sentinel-2")
    if es_s2:
        nombre = _nombre_catalogo(catalogo)
    else:
        if catalogo is not None and _nombre_catalogo(catalogo) != "planetary":
            raise ValueError("Landsat solo está en Planetary Computer: usa conectar_catalogo('planetary').")
        nombre = "planetary"

    if nombre == "planetary":
        nativas = list(bandas)
    else:
        nativas = [BANDAS_EARTHSEARCH.get(banda, banda) for banda in bandas]

    configurar_lectura()
    parametros = {"bands": nativas, "crs": crs, "resolution": resolucion}
    if bbox is not None:
        parametros["bbox"] = bbox
    if chunks:
        parametros["chunks"] = chunks
    parametros["groupby"] = agrupar_por_dia_y_offset if es_s2 else "solar_day"
    if nombre == "planetary":
        parametros["patch_url"] = firmar_url
    return parametros


def renombrar_bandas(datos, catalogo=None):
    """
    Deja las bandas de Sentinel-2 con los nombres de la ESA (B02, B04, B08, SCL, ...).

    Planetary Computer ya usa esos nombres (no cambia nada). Earth Search nombra
    sus archivos "blue", "red", "nir", "scl", ...; se traducen según
    `BANDAS_EARTHSEARCH`. Solo para Sentinel-2 (no uses esta función con Landsat).

    Args:
        datos (xarray.Dataset): Resultado de `odc.stac.load`.
        catalogo (str o pystac_client.Client, optional): Catálogo de donde vienen
            los datos. Si es None se usa `CATALOGO_SENTINEL2`.

    Returns:
        xarray.Dataset: Mismos datos con las bandas renombradas.
    """
    if _nombre_catalogo(catalogo) == "planetary":
        return datos
    inverso = {asset: banda for banda, asset in BANDAS_EARTHSEARCH.items()}
    return datos.rename({nombre: inverso[nombre] for nombre in datos.data_vars if nombre in inverso})


def agregar_offset(datos, items):
    """
    Agrega a `datos` la coordenada `offset_reflectancia` (un valor por fecha).

    Es lo que luego usa `utils.raster.convertir_a_reflectancia` para aplicar el
    offset correcto de cada imagen (ver `offset_sentinel2`). Debes pasar los mismos
    `items` (ya con `preparar_items`) que usaste en `odc.stac.load` y haber cargado
    con `parametros_carga`, cuyo `groupby` asegura que cada fecha tenga un solo offset.
    Si los ítems no son Sentinel-2 (Landsat), devuelve `datos` sin cambios.

    Args:
        datos (xarray.Dataset): Resultado de `odc.stac.load`.
        items (iterable): Ítems usados en la carga.

    Returns:
        xarray.Dataset: `datos`, ordenado por fecha, con la coordenada
        `offset_reflectancia`.

    Raises:
        ValueError: Si hay catálogos mezclados, versiones repetidas de una misma
            toma (usa `preparar_items`) o fechas de `datos` que no corresponden
            a los ítems.
    """
    items = list(items)
    if not items:
        raise ValueError("No hay imágenes para asociar con los datos.")
    if not any(es_sentinel2(item) for item in items):
        return datos  # Landsat: el offset es fijo (-0.2) y lo conoce convertir_a_reflectancia
    if not all(es_sentinel2(item) for item in items):
        raise ValueError("No se pueden mezclar Sentinel-2 con otras colecciones en una carga.")
    nombre = _catalogo_de_items(items)
    if len(quitar_duplicados(items)) != len(items):
        raise ValueError("Hay versiones repetidas de una misma toma Sentinel-2: pasa los ítems por preparar_items().")

    # Misma agrupación que hizo odc.stac: la fecha de cada grupo es la del primer ítem.
    fecha_de_grupo = {}
    for item in items:
        marca = np.datetime64(item.datetime.replace(tzinfo=None), "ns")
        clave = agrupar_por_dia_y_offset(item)
        fecha_de_grupo[clave] = min(marca, fecha_de_grupo.get(clave, marca))
    offset_de_fecha = {}
    for (_, offset), marca in fecha_de_grupo.items():
        if marca in offset_de_fecha:
            raise ValueError(f"Dos grupos de imágenes tienen la misma fecha exacta ({marca}); no se puede asignar el offset.")
        offset_de_fecha[marca] = offset

    # Un día con varios offsets debe verse completo en `datos`: si solo aparece uno de sus
    # grupos, la carga mezcló imágenes con offsets distintos (no se usó el `groupby`
    # de parametros_carga) y el offset asignado sería falso para parte de ellas.
    tiempos = set(datos["time"].values.astype("datetime64[ns]"))
    por_dia = {}
    for (dia, _), marca in fecha_de_grupo.items():
        por_dia.setdefault(dia, []).append(marca)
    for dia, marcas in por_dia.items():
        presentes = sum(marca in tiempos for marca in marcas)
        if len(marcas) > 1 and 0 < presentes < len(marcas):
            raise ValueError(
                f"El {dia} hay imágenes con offsets de reflectancia distintos que quedaron en una sola "
                "fecha. Carga con odc.stac.load(items, **parametros_carga(...)): su groupby las separa."
            )

    try:
        offsets = [offset_de_fecha[marca] for marca in datos["time"].values.astype("datetime64[ns]")]
    except KeyError as error:
        raise ValueError(
            f"Las fechas de los datos no corresponden a los ítems (falta {error.args[0]}). "
            "Carga con odc.stac.load(items, **parametros_carga(...)) usando los mismos ítems."
        ) from None
    datos = datos.assign_coords({COORD_OFFSET: ("time", np.array(offsets, dtype=float))})
    datos = datos.sortby("time")  # dos grupos del mismo día podrían quedar desordenados
    datos[COORD_OFFSET].attrs = {
        "long_name": "offset de reflectancia de cada fecha",
        "description": "reflectancia = DN * 0.0001 + offset_reflectancia",
        "catalogo": nombre,
    }
    return datos


# ---------------------------------------------------------------------------
# Otros
# ---------------------------------------------------------------------------
def cargar_dem(bbox, epsg="EPSG:32719", resolucion=30, catalogo=None):
    """
    Carga el modelo digital de elevación Copernicus (30 m) del área.

    Args:
        bbox (list): [oeste, sur, este, norte] en grados.
        epsg (str): CRS de salida.
        resolucion (int): Tamaño de píxel en metros.
        catalogo (pystac_client.Client, optional): Cliente de Planetary Computer.
            Si no se entrega (o es de otro catálogo), se conecta uno.

    Returns:
        xarray.DataArray: Elevación en metros (dimensiones y, x), o None si no hay datos.
    """
    if catalogo is None or isinstance(catalogo, str) or "planetarycomputer" not in (catalogo.get_self_href() or ""):
        catalogo = conectar_catalogo("planetary")
    items = list(catalogo.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    if not items:
        print("No se encontraron datos de elevación para el área.")
        return None
    dem = odc.stac.load(items, bbox=bbox, crs=epsg, resolution=resolucion)
    return dem["data"].squeeze()


def escala_y_offset(item, banda):
    """
    Factor de escala y offset para pasar de DN a reflectancia en una banda.

    * Sentinel-2: se devuelve el valor correcto para ese ítem (0.0001 y el
      offset de `offset_sentinel2`), sin importar lo que digan los metadatos.
    * Landsat y otros: se leen de `raster:bands` en los metadatos STAC.

    Args:
        item (pystac.Item): Una imagen encontrada en el catálogo.
        banda (str): Nombre de la banda, por ejemplo "red" (Landsat) o "B04".

    Returns:
        tuple: (scale, offset). Si el ítem no los informa, (1.0, 0.0).
    """
    if es_sentinel2(item):
        return 0.0001, offset_sentinel2(item)
    info = item.assets[banda].extra_fields.get("raster:bands", [{}])[0]
    return info.get("scale", 1.0), info.get("offset", 0.0)
