# %% [markdown]
# # Bloque 2. Catálogos STAC y búsqueda de imágenes
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/02_stac_busqueda.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# Taller *Datos FAIR, series temporales satelitales y detección de anomalías con Python*. Día 1, mañana (unos 60 minutos).
#
# En el bloque anterior vimos que un dato se encuentra solo si está bien descrito. Los **catálogos STAC** son esa idea aplicada a las imágenes satelitales: cada imagen viene con metadatos estándar (fecha, nubosidad, huella, versión de procesamiento) y un programa puede buscarlas por esos metadatos, sin descargar nada.
#
# **Qué vas a hacer**
#
# | Parte | Tema | Minutos |
# |---|---|---|
# | 1 | Conceptos: catálogo, colección, ítem y asset | 15 |
# | 2 | La zona de estudio y su *bounding box* | 5 |
# | 3 | Buscar Sentinel-2 con `pystac_client` (llamada nativa) | 10 |
# | 4 | Inspeccionar los metadatos: fecha, nubes, huella, baseline y tile | 15 |
# | 5 | Duplicados por reprocesamiento y `preparar_items` | 5 |
# | 6 | El offset de reflectancia depende del catálogo y del baseline | 5 |
# | 7 | Un caso de enseñanza: la escena con archivos sin CRS | 5 |
#
# En todo el taller seguimos el **patrón nativo** del curso: las llamadas a `pystac_client` y `odc.stac` las escribes tú. Los helpers de `utils/` solo **preparan** sus argumentos y ordenan los datos antes y después.

# %% [markdown]
# ## Preparar el entorno
#
# La celda `Preparar entorno` clona el repositorio del curso (para usar `utils/`). La siguiente instala los paquetes de STAC que Colab no trae, solo si faltan.

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (y actualizarlo si ya estaba)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
    else:  # una sesión anterior ya lo clonó: traer lo nuevo para no usar código viejo
        subprocess.run(["git", "-C", str(RAIZ), "pull", "-q", "--ff-only"], check=False)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalar los paquetes de este bloque (solo si faltan)
import importlib.util

# Paquete de requirements.txt -> nombre con el que se importa
NECESARIOS = {"pystac-client": "pystac_client", "planetary-computer": "planetary_computer",
              "odc-stac": "odc.stac", "rioxarray": "rioxarray"}
requisitos = [linea.strip() for linea in (RAIZ / "talleres/fair-series-temporales/requirements.txt").read_text().splitlines()
              if linea.strip() and not linea.startswith("#")]
requisitos = [r for r in requisitos if r.split()[0] in NECESARIOS]

if any(importlib.util.find_spec(modulo) is None for modulo in NECESARIOS.values()):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *requisitos], check=True)
    print("Instalado:", ", ".join(requisitos))
else:
    print("Ya está todo instalado:", ", ".join(NECESARIOS))

# %%
# @title Importar bibliotecas
import warnings

import folium
import matplotlib.pyplot as plt
import odc.stac
import pandas as pd
import planetary_computer
import rasterio
from pystac_client import Client

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import convertir_a_reflectancia
from utils.series import bbox_punto, bbox_puntos, cargar_puntos_csv, crs_utm
from utils.stac import (agregar_offset, conectar_catalogo, offset_sentinel2, parametros_busqueda, parametros_carga,
                        preparar_items, quitar_duplicados, resumir_imagenes)
from utils.viz import mapa_puntos

warnings.filterwarnings("ignore", message="Dataset has no geotransform")  # ruido de GDAL con la escena rota (parte 7)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

# %% [markdown]
# ## 1. Conceptos: catálogo, colección, ítem y asset
#
# **STAC** (*SpatioTemporal Asset Catalog*) es un estándar para describir datos geoespaciales con la misma estructura, sin importar quién los publique. Tiene cuatro piezas, de la más general a la más concreta:
#
# ```text
#   Catálogo            Planetary Computer (la "biblioteca")
#     └─ Colección      sentinel-2-l2a  (un producto: Sentinel-2 nivel 2A)
#          └─ Ítem      S2C_MSIL2A_20260810T143741_R096_T18HXE_...  (una toma: una fecha, un tile)
#               └─ Asset   B04.tif, B08.tif, SCL.tif, ...  (los archivos: una banda por archivo)
# ```
#
# | Pieza | Qué es | Ejemplo |
# |---|---|---|
# | **Catálogo** | Un servicio con muchas colecciones y una API de búsqueda | `https://planetarycomputer.microsoft.com/api/stac/v1` |
# | **Colección** | Un producto con las mismas propiedades (cobertura, licencia, resolución) | `sentinel-2-l2a` |
# | **Ítem** | Una observación: identificador, fecha, huella geográfica y propiedades (nubes, baseline...) | una imagen del 10 de agosto de 2026 sobre el tile 18HXE |
# | **Asset** | Un archivo del ítem, con su dirección (`href`) | la banda roja `B04` como GeoTIFF en la nube |
#
# Lo importante es que **la búsqueda ocurre sobre metadatos**: le dices al catálogo "dame los ítems de esta colección, que toquen este rectángulo, en estas fechas, con menos de 30 % de nubes", y te devuelve solo la lista de ítems. Los archivos pesados se leen después, y solo la parte que necesitas (lo vemos en el bloque 3).
#
# Es el mismo tema del bloque 1: los campos de STAC (`datetime`, `eo:cloud_cover`, `geometry`, `s2:processing_baseline`...) son **metadatos FAIR en acción**. Sin ellos no habría forma de buscar.

# %% [markdown]
# ## 2. La zona de estudio y su bounding box
#
# El taller trabaja sobre el agua y la costa de Concepción, Talcahuano y el golfo de Arauco. Los puntos de muestreo están en un CSV (`datos/puntos_concepcion.csv`). `cargar_puntos_csv()` lo lee y valida que las columnas y las coordenadas tengan sentido (un error clásico es invertir latitud y longitud).

# %%
puntos = cargar_puntos_csv()  # puntos_concepcion.csv del taller
puntos

# %% [markdown]
# Para buscar imágenes necesitamos un rectángulo (*bounding box*, `bbox`) en grados: `[oeste, sur, este, norte]`. `bbox_puntos` calcula el que contiene las ventanas de 150 m de radio de **todos** los puntos, así hacemos una sola búsqueda. Lo dibujamos junto con los puntos.

# %%
bbox = bbox_puntos(puntos, radio_m=150)
print("bbox [oeste, sur, este, norte]:", [round(v, 4) for v in bbox])

# %%
mapa = mapa_puntos(puntos, radio_m=150)
folium.Rectangle(bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color="red", fill=False, weight=2,
                 tooltip="bbox de la búsqueda").add_to(mapa)
mapa

# %% [markdown]
# ## 3. Buscar Sentinel-2 con `pystac_client`
#
# Conectamos con el catálogo de **Microsoft Planetary Computer** con la llamada nativa de `pystac_client`. El parámetro `modifier=planetary_computer.sign_inplace` firma los enlaces de los archivos para poder leerlos (las firmas caducan al cabo de un rato, ver el bloque 4).

# %%
catalogo = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
print(catalogo.title)

# %% [markdown]
# Un catálogo tiene muchas **colecciones**. Veamos cuántas y cómo se describe la de Sentinel-2 nivel 2A (reflectancia de superficie).

# %%
# Para listar usamos un cliente sin firma: firmar cada colección falla (403) en algunas de ellas.
colecciones = list(Client.open("https://planetarycomputer.microsoft.com/api/stac/v1").get_collections())
print(f"Planetary Computer tiene {len(colecciones)} colecciones. Algunas:")
for coleccion in colecciones[:8]:
    print(f"  {coleccion.id:<35} {coleccion.title}")

sentinel2 = catalogo.get_collection("sentinel-2-l2a")
print("\nColección:", sentinel2.id, "-", sentinel2.title)
print("Licencia:", sentinel2.license)
print("Cobertura temporal:", sentinel2.extent.temporal.intervals[0][0].date(), "a", sentinel2.extent.temporal.intervals[0][1])

# %% [markdown]
# Ahora la búsqueda sobre la zona de estudio: imágenes de **enero a marzo de 2026** con **menos de 30 % de nubes** en la escena. Es la llamada nativa `catalogo.search(...)`: los argumentos son la colección, el rectángulo, el rango de fechas y un filtro sobre la propiedad `eo:cloud_cover`.

# %%
busqueda_nativa = dict(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2026-01-01/2026-03-31",
    query={"eo:cloud_cover": {"lt": 30}},
)
busqueda = catalogo.search(**busqueda_nativa)
items = busqueda.item_collection()
print(f"El catálogo devolvió {len(items)} ítems")

# %% [markdown]
# ### El helper `parametros_busqueda`
#
# Armar ese diccionario a mano es fácil, pero se repite en cada cuaderno. `parametros_busqueda` lo construye por ti. **No busca nada**: devuelve el diccionario que tú le pasas a `catalogo.search(**...)`. Comprobemos que es exactamente el mismo.

# %%
parametros = parametros_busqueda("sentinel-2", bbox, ("2026-01-01", "2026-03-31"), nubes_max=30, catalogo=catalogo)
print(parametros)
print("¿Es igual al diccionario que escribimos a mano?", parametros == busqueda_nativa)

# %% [markdown]
# **Tu turno (3 min).** Cambia `nubes_max` a 10 y las fechas a otro trimestre (por ejemplo, de abril a junio). ¿Cuántos ítems devuelve ahora? ¿Por qué en invierno hay menos imágenes con pocas nubes?

# %%
# Tu turno: otra búsqueda con la llamada nativa (usa el diccionario que arma el helper)
mis_parametros = parametros_busqueda("sentinel-2", bbox, ("2026-04-01", "2026-06-30"), nubes_max=10, catalogo=catalogo)
mis_items = catalogo.search(**mis_parametros).item_collection()
print(f"{len(mis_items)} ítems con menos de 10 % de nubes entre abril y junio de 2026")

# %% [markdown]
# ## 4. Inspeccionar los metadatos
#
# `items` es una colección de ítems STAC. Miremos uno en detalle: sus propiedades son los metadatos que usamos para decidir qué imagen sirve.

# %%
item = items[0]
print("id:       ", item.id)
print("fecha:    ", item.datetime)
print("nubes (%):", round(item.properties["eo:cloud_cover"], 2))
print("tile:     ", item.properties["s2:mgrs_tile"])
print("baseline: ", item.properties["s2:processing_baseline"])
print("satélite: ", item.properties["platform"])
print("colección:", item.collection_id)

# %% [markdown]
# * **`id`**: el nombre de la toma. Lleva el satélite (`S2A`, `S2B`, `S2C`), el nivel de procesamiento, la fecha y hora de adquisición, la órbita relativa, el **tile** (`T18HXE`) y la fecha de procesamiento.
# * **`tile`**: Sentinel-2 se publica en teselas de 100 x 100 km del sistema MGRS. Nuestra zona cae en `18HXE` (norte) y `18HXD` (sur).
# * **`s2:processing_baseline`**: la versión del procesador de la ESA. Importa más de lo que parece (partes 5 y 6).
#
# Los **assets** son los archivos del ítem. Mostramos los primeros sin su firma (la parte después del `?` es una credencial temporal que no se muestra).

# %%
print(f"{'asset':<14} {'resolución':>10}   tipo / dirección (sin firma)")
for nombre, asset in list(item.assets.items())[:16]:
    gsd = asset.extra_fields.get("gsd", "")
    print(f"{nombre:<14} {str(gsd):>10}   {(asset.media_type or '').split(';')[0]:<28} {asset.href.split('?')[0][-50:]}")

# %% [markdown]
# Las bandas espectrales (`B02`, `B03`, `B04`, `B08`...) son GeoTIFF en la nube y `SCL` es la clasificación de la escena (nubes, sombras, agua...). `visual` es una imagen RGB ya lista y `rendered_preview` una miniatura. Verás los archivos en acción en el bloque 3.
#
# ### Una tabla con lo importante
#
# `resumir_imagenes` entrega una tabla con fecha, nubes, id, baseline y el offset de reflectancia de cada imagen. Agregamos el tile desde las propiedades para ver qué cubre cada una.

# %%
tabla = resumir_imagenes(items)
tabla["tile"] = tabla["id"].str.split("_").str[4].str[1:]  # el tile está dentro del id: ..._R096_T18HXE_...
tabla.head(10)

# %% [markdown]
# Una misma fecha aparece varias veces: los dos tiles (`18HXD` y `18HXE`) y a veces dos satélites o dos órbitas. **Eso no son duplicados**, son tomas distintas. Los duplicados verdaderos los vemos en la parte 5.
#
# Veamos cuántas tomas hay por fecha y cómo varía la nubosidad a lo largo del trimestre.

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 3.8))
fechas = pd.to_datetime(tabla["fecha"])
for tile, grupo in tabla.groupby("tile"):
    ejes[0].scatter(pd.to_datetime(grupo["fecha"]), grupo["nubes_%"], label=tile, s=25)
ejes[0].set_title("Nubosidad de la escena por fecha")
ejes[0].set_ylabel("Nubes (%)")
ejes[0].legend(title="tile")
ejes[0].tick_params(axis="x", rotation=30)
tabla["fecha"].value_counts().sort_index().plot.bar(ax=ejes[1], width=0.8)
ejes[1].set_title("Ítems por fecha")
ejes[1].set_xticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### La huella de cada imagen
#
# Cada ítem trae su **huella** (`geometry`): el polígono que cubre. Dibujamos las de los dos tiles junto con los puntos de muestreo. Fíjate en que los tiles se **traslapan** y en que un punto puede quedar cubierto por un tile y no por el otro.

# %%
mapa = mapa_puntos(puntos, radio_m=150)
for tile, color in {"18HXE": "#3388ff", "18HXD": "#ff8833"}.items():
    ejemplo = next(i for i in items if i.properties["s2:mgrs_tile"] == tile)
    folium.GeoJson(ejemplo.geometry, name=f"huella {tile}",
                   style_function=lambda _, color=color: {"color": color, "weight": 2, "fillOpacity": 0.05},
                   tooltip=f"tile {tile}").add_to(mapa)
folium.Rectangle(bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color="red", fill=False, weight=2).add_to(mapa)
mapa

# %% [markdown]
# **Tu turno (4 min).**
#
# 1. Filtra la tabla para quedarte con el tile `18HXE` y menos de 5 % de nubes. ¿Cuántas fechas quedan?
# 2. ¿Cuál es la imagen más despejada del trimestre? ¿Y de qué satélite es?
# 3. ¿Qué puntos del CSV quedarían fuera si solo usaras el tile `18HXE`? (Pista: mira el mapa.)

# %%
# Tu turno
despejadas = tabla[(tabla["tile"] == "18HXE") & (tabla["nubes_%"] < 5)]
print(f"{len(despejadas)} tomas del tile 18HXE con menos de 5 % de nubes")
mejor = despejadas.sort_values("nubes_%").iloc[0]
print("La más despejada:", mejor["id"], f"({mejor['nubes_%']:.3f} % de nubes)")

# %% [markdown]
# ## 5. Duplicados por reprocesamiento y `preparar_items`
#
# La ESA a veces **reprocesa** imágenes antiguas con una versión nueva del procesador. El catálogo entonces guarda las dos versiones de la misma toma, con el mismo instante de adquisición pero distinta fecha de procesamiento y distinto `baseline`. Si cargamos ambas, la fecha cuenta doble.
#
# Veámoslo con datos reales. Buscamos entre el 20 de enero y el 10 de febrero de **2022**, un período con varios reprocesamientos.

# %%
parametros_2022 = parametros_busqueda("sentinel-2", bbox, ("2022-01-20", "2022-02-10"), nubes_max=50, catalogo=catalogo)
items_2022 = catalogo.search(**parametros_2022).item_collection()
print(f"El catálogo devolvió {len(items_2022)} ítems")

tabla_2022 = resumir_imagenes(items_2022)
# Una toma = satélite + instante de adquisición + tile (partes del id separadas por "_")
partes = tabla_2022["id"].str.split("_")
tabla_2022["toma"] = partes.str[0] + "_" + partes.str[2] + "_" + partes.str[4]
repetidas = tabla_2022[tabla_2022.duplicated("toma", keep=False)]
repetidas[["fecha", "id", "baseline", "offset"]]

# %% [markdown]
# La misma toma del **5 de febrero de 2022** aparece dos veces en cada tile: una procesada en febrero de 2022 (baseline `04.00`) y otra reprocesada en **mayo de 2024** (baseline `05.10`). Es la misma imagen, con la corrección de la versión nueva.
#
# `preparar_items` limpia los resultados antes de cargar: **quita las versiones repetidas** (deja la de baseline más alto), exige que todos los ítems vengan de **un solo catálogo** y los ordena por fecha. Sigue siendo tu código el que busca y carga; el helper solo ordena lo que encontraste.

# %%
items_2022_limpios = preparar_items(items_2022)
print("Antes:", len(items_2022), "ítems | Después:", len(items_2022_limpios), "ítems")
quedan = resumir_imagenes(items_2022_limpios)
quedan[quedan["fecha"] == "2022-02-05"][["fecha", "id", "baseline", "offset"]]

# %% [markdown]
# Quedó la versión `05.10`. Con los datos de 2026 que buscamos antes no había duplicados (todo está procesado con baseline `05.11` a `05.13`), pero **siempre** pasamos los ítems por `preparar_items`, porque no sabes qué reprocesamientos tiene el período que vas a pedir.
#
# **Tu turno (2 min).** `quitar_duplicados(items_2022, verbose=True)` hace solo la parte de los duplicados. Pruébalo, y comprueba que los ítems de una misma fecha pero de distinto tile **no** se consideran duplicados.

# %%
# Tu turno
sin_duplicados = quitar_duplicados(items_2022, verbose=True)
print("Fechas distintas:", len({i.datetime.date() for i in sin_duplicados}), "| ítems:", len(sin_duplicados))

# %% [markdown]
# ## 6. El offset de reflectancia depende del catálogo y del baseline
#
# Los píxeles de Sentinel-2 L2A vienen como **números digitales** (DN) y se convierten a reflectancia con `reflectancia = DN * 0.0001 + offset`. Desde el baseline `04.00` (introducido el 25 de enero de 2022) la ESA **suma 1000 al DN** para poder guardar reflectancias negativas, así que hay que restar `0.1`:
#
# * baseline `< 04.00`: `reflectancia = DN * 0.0001`
# * baseline `>= 04.00`: `reflectancia = DN * 0.0001 - 0.1`
#
# Pero **no basta con mirar la fecha**: la ESA reprocesó imágenes antiguas, y cada catálogo entrega los DN a su manera. Por eso `offset_sentinel2(item)` decide el offset **de cada ítem** según el catálogo y el baseline. Lo usamos sobre ítems reales del período de 2022 que acabamos de buscar. Fíjate en lo que pasa alrededor del 26 de enero de 2022.

# %%
filas = []
for it in items_2022_limpios:
    filas.append({"fecha": it.datetime.strftime("%Y-%m-%d"), "baseline": it.properties["s2:processing_baseline"],
                  "offset_sentinel2": offset_sentinel2(it)})
cambio = pd.DataFrame(filas).drop_duplicates().reset_index(drop=True)
cambio

# %% [markdown]
# Hasta el 23 de enero las imágenes son baseline `03.00` (offset `0`) y desde el 26 de enero `04.00` (offset `-0.1`). Y las de 2026 son todas `05.11` a `05.13`, con offset `-0.1`.
#
# ### ¿Y si aplico un valor fijo?
#
# Veamos qué pasa con **agua**, donde la reflectancia es muy baja y un error de 0.1 se nota más. Cargamos una ventana de unos 600 x 600 m en la Bahía de Concepción para cuatro fechas de ese período, con el patrón nativo: `parametros_carga` + `odc.stac.load` + `agregar_offset` (los detalles de la carga son del bloque 3).

# %%
bahia = puntos.set_index("id").loc["bahia_concepcion"]
ventana = bbox_punto(float(bahia.lon), float(bahia.lat), 300)

parametros_2022b = parametros_busqueda("sentinel-2", ventana, ("2022-01-20", "2022-02-10"), nubes_max=20, catalogo=catalogo)
items_ventana = preparar_items(catalogo.search(**parametros_2022b).item_collection())

datos = odc.stac.load(items_ventana, **parametros_carga("sentinel-2", ["B04"], resolucion=10,
                                                         crs=crs_utm(float(bahia.lon), float(bahia.lat)),
                                                         catalogo=catalogo, bbox=ventana))
datos = agregar_offset(datos, items_ventana)  # el offset correcto de cada fecha

dn_mediano = datos["B04"].median(["x", "y"])  # DN mediano de la ventana, por fecha
comparacion = pd.DataFrame({
    "fecha": pd.to_datetime(datos.time.values).strftime("%Y-%m-%d"),
    "baseline": [i.properties["s2:processing_baseline"] for i in sorted(items_ventana, key=lambda i: i.datetime)],
    "DN B04 (mediana)": dn_mediano.values,
    "offset correcto": datos.offset_reflectancia.values,
})
comparacion["reflectancia correcta"] = dn_mediano.values * 0.0001 + comparacion["offset correcto"]
comparacion["con -0.1 fijo"] = dn_mediano.values * 0.0001 - 0.1
comparacion.round(4)

# %% [markdown]
# El agua de la bahía es la misma, pero el **DN salta** de unas decenas o centenas (baseline `03.00`) a más de 1.100 (baseline `04.00` y posteriores): son los 1000 que se sumaron. Con el offset de cada fecha la reflectancia roja del agua queda coherente (entre 0.004 y 0.034, típica de agua oscura). Con un `-0.1` fijo, las dos fechas anteriores a `04.00` darían **reflectancias negativas** (imposibles, y la función de conversión las pondría en `NaN`).
#
# La conversión del curso usa el offset de cada fecha automáticamente:

# %%
reflectancia = convertir_a_reflectancia(datos[["B04"]], "sentinel-2")  # usa la coordenada offset_reflectancia
reflectancia["B04"].median(["x", "y"]).to_series().round(4)

# %% [markdown]
# **Por qué nunca fijamos un valor a mano**
#
# * Depende del **baseline**, no de la fecha: hay tomas de diciembre de 2021 reprocesadas con `04.00`.
# * Depende del **catálogo**: en Planetary Computer decide el baseline; Earth Search Colección 1 (`sentinel-2-c1-l2a`) siempre lleva `-0.1`, porque todo está reprocesado.
# * Un error de offset **no da error**: da una serie que se ve plausible pero está desplazada en 0.1, y los índices (NDVI, NDWI) cambian.
#
# | Catálogo | Colección | Offset |
# |---|---|---|
# | Planetary Computer (por defecto) | `sentinel-2-l2a` | `0` si baseline `< 04.00`; `-0.1` si `>= 04.00` |
# | Earth Search (AWS) | `sentinel-2-c1-l2a` | siempre `-0.1` |
#
# Los helpers del curso (`offset_sentinel2`, `agregar_offset`, `convertir_a_reflectancia`) aplican esta regla por imagen, y tienen pruebas automáticas con metadatos y píxeles reales de la zona. **Los cuadernos antiguos con un offset fijo no deben copiarse.** La explicación completa está en `help(utils.stac)`.

# %% [markdown]
# ## 7. Un caso de enseñanza: la escena con archivos sin CRS
#
# Los metadatos de un catálogo son una promesa sobre los archivos, y a veces la promesa falla. En Planetary Computer, la escena
#
# `S2B_MSIL2A_20260407T143739_R096_T18HXD_20260407T180947`
#
# declara en su ítem el sistema de coordenadas `EPSG:32718`, pero **sus archivos de banda no lo traen**. GDAL no sabe georreferenciarlos y hace fallar toda la carga, aunque las demás escenas estén bien. Veamos cómo se detecta.
#
# Primero, lo que dicen los metadatos del ítem, y lo que dicen los archivos. `rasterio.open` lee solo el encabezado del archivo (no descarga la imagen), así que es barato.

# %%
ID_ROTA = "S2B_MSIL2A_20260407T143739_R096_T18HXD_20260407T180947"
ID_SANA = "S2B_MSIL2A_20260407T143739_R096_T18HXE_20260407T180947"  # el tile vecino, la misma toma


def crs_del_archivo(item, banda="B04"):
    """CRS que trae el archivo de la banda (None si no lo trae). Solo lee el encabezado."""
    with rasterio.open(item.assets[banda].href) as archivo:
        return archivo.crs


coleccion = catalogo.get_collection("sentinel-2-l2a")
for id_item in (ID_SANA, ID_ROTA):
    it = coleccion.get_item(id_item)
    print(f"{id_item}\n   metadatos del ítem: {it.properties.get('proj:code')} | archivo B04: {crs_del_archivo(it)}")

# %% [markdown]
# La escena sana trae `EPSG:32718` en ambos lados. En la rota **el ítem dice `EPSG:32718` pero el archivo dice `None`**. Esa diferencia es la señal. Mira qué pasa si intentamos cargar la escena rota con otras sanas del mismo día (usamos una ventana pequeña en el golfo de Arauco, que cae en el tile `18HXD`).

# %%
arauco = puntos.set_index("id").loc["golfo_arauco"]
ventana_arauco = bbox_punto(float(arauco.lon), float(arauco.lat), 150)
parametros_dia = parametros_busqueda("sentinel-2", ventana_arauco, ("2026-04-07", "2026-04-07"), nubes_max=100, catalogo=catalogo)
items_dia = preparar_items(catalogo.search(**parametros_dia).item_collection())

try:
    odc.stac.load(items_dia, **parametros_carga("sentinel-2", ["B04"], resolucion=10, crs=crs_utm(float(arauco.lon), float(arauco.lat)),
                                                catalogo=catalogo, bbox=ventana_arauco))
    print("La carga funcionó")
except Exception as error:  # noqa: BLE001
    print(f"La carga falló con {type(error).__name__}: '{error}'")
    print("(Un error sin mensaje: odc.stac no dice qué escena ni qué archivo causó el problema.)")

# %% [markdown]
# El error es un `AssertionError` **sin mensaje**: no dice qué escena lo causó. Cuando una carga de muchas fechas falla así, la causa suele ser **un solo archivo defectuoso**. Para encontrarlo, comprobamos cada ítem con `crs_del_archivo` **antes** de cargar y descartamos los que no pasan.

# %%
sin_crs = [i.id for i in items_dia if crs_del_archivo(i) is None]
print("Ítems con archivos sin CRS:", sin_crs)

items_buenos = [i for i in items_dia if i.id not in sin_crs]
datos_ok = odc.stac.load(items_buenos, **parametros_carga("sentinel-2", ["B04"], resolucion=10, crs=crs_utm(float(arauco.lon), float(arauco.lat)),
                                                         catalogo=catalogo, bbox=ventana_arauco))
print(f"Sin la escena defectuosa, la carga funciona: {datos_ok.sizes['time']} fecha(s), ventana de {datos_ok.sizes['y']} x {datos_ok.sizes['x']} píxeles")

# %% [markdown]
# **Qué aprendemos de esto**
#
# * Los metadatos pueden estar bien y el archivo mal (o al revés). **Valida** lo que vas a usar, sobre todo antes de una carga larga.
# * Leer el encabezado de un archivo COG es muy barato: revisar el CRS de cada ítem toma un par de segundos.
# * En el taller, la serie temporal del bloque 4 **excluye esta escena por su id** (`ITEMS_DEFECTUOSOS`).
# * Es un buen caso para **reportar** al proveedor del catálogo, con el id de la escena y la evidencia.
#
# **Tu turno (3 min).** Revisa el CRS del archivo `SCL` de la escena rota (no solo el de `B04`). ¿Pasa lo mismo? Después, escribe una función `item_ok(item)` que devuelva `True` solo si el CRS del archivo coincide con `proj:code` del ítem.

# %%
# Tu turno
print("SCL de la escena rota:", crs_del_archivo(coleccion.get_item(ID_ROTA), banda="SCL"))


def item_ok(item, banda="B04"):
    """True si el CRS del archivo coincide con el que declara el ítem."""
    crs = crs_del_archivo(item, banda)
    return crs is not None and crs.to_string() == item.properties.get("proj:code")


print({i.id[-30:]: item_ok(i) for i in items_dia})

# %% [markdown]
# ## Resumen
#
# * Un **catálogo STAC** organiza los datos en colecciones, ítems y assets, y se busca por metadatos (zona, fechas, nubes) sin descargar nada.
# * La búsqueda es la llamada nativa `catalogo.search(**parametros)`. `parametros_busqueda` solo arma el diccionario, y `preparar_items` limpia el resultado.
# * Cada ítem informa **fecha, nubes, huella, tile y baseline**. Los tiles se traslapan y una fecha puede tener varias tomas legítimas.
# * Los **duplicados** son reprocesamientos de la misma toma con otro baseline. Se dejan con `preparar_items`.
# * El **offset** de reflectancia (`0` o `-0.1`) depende del catálogo y del baseline: se decide por imagen con `offset_sentinel2`, nunca con un valor fijo.
# * Un metadato correcto no garantiza un archivo correcto: **valida** antes de cargar.
#
# Siguiente bloque: **leer un COG de Sentinel-2** y calcular imágenes e índices del agua (`03_acceso_sentinel2`).
