# %% [markdown]
# # Acceso a Imágenes Satelitales
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/03_acceso_imagenes.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# En este cuaderno aprenderemos a acceder y visualizar imágenes satelitales utilizando Planetary Computer.

# %% [markdown]
# ## Configuración del entorno
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (buscar imágenes, convertir a reflectancia, mostrar imágenes, etc.).

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
# @title Instalación de paquetes necesarios
# Necesitamos instalar las librerías necesarias para trabajar con imágenes satelitales
# %pip install -q rioxarray xarray matplotlib numpy rasterio xarray-spatial geopandas planetary-computer pystac-client odc-stac

# %%
# @title Importación de bibliotecas
import odc.stac
import planetary_computer
from pystac_client import Client

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import convertir_a_reflectancia
from utils.stac import agregar_offset, escala_y_offset, parametros_busqueda, parametros_carga, preparar_items
from utils.viz import crear_imagen_rgb, mostrar_imagen

# %% [markdown]
# ## 1. Introducción a Planetary Computer
#
# Microsoft Planetary Computer es una plataforma que proporciona acceso a datos ambientales y climáticos, incluyendo:
#
# * Imágenes satelitales (Landsat, Sentinel)
# * Datos climáticos
# * Datos de elevación
# * Datos de cobertura terrestre
#
# Ventajas:
# * Acceso gratuito
# * Datos listos para usar
# * API fácil de usar
# * Integración con herramientas de análisis

# %% [markdown]
# Planetary Computer organiza sus datos en un **catálogo STAC**. Para usarlo nos conectamos con `Client.open(...)`; el parámetro `modifier=planetary_computer.sign_inplace` firma automáticamente los enlaces para que podamos descargar las imágenes.
#
# > **Nota:** en los demás cuadernos usaremos `conectar_catalogo()` de `utils.stac`, que hace exactamente lo que ves en esta sección. Más adelante veremos otros helpers de ese módulo (`parametros_busqueda()`, `parametros_carga()`...): no reemplazan estas llamadas, solo **preparan** sus argumentos y ordenan los datos.

# %%
# Conectamos al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# %% [markdown]
# ## 2. Búsqueda de Imágenes Sentinel-2
#
# Vamos a buscar imágenes Sentinel-2 para un área de interés en Chile.

# %%
# Definimos el área de interés
bbox = [-71.399460, -34.366111, -70.633850, -34.084512]

# Definimos el período de tiempo
fecha_inicio = "2023-01-01"
fecha_fin = "2023-01-31"
time_range = f"{fecha_inicio}/{fecha_fin}"

# Realizamos la búsqueda
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 20}},  # Menos de 20% de nubes
)

# Recolectamos los resultados de la búsqueda como una lista de ítems STAC
items = search.item_collection()
print(f"Encontradas {len(items)} imágenes")

# Mostramos información de la primera imagen
item = items[0]
print("\nInformación de la primera imagen:")
print(f"Fecha: {item.datetime.strftime('%Y-%m-%d')}")
print(f"Cobertura de nubes: {item.properties['eo:cloud_cover']}%")
print(f"ID de la imagen: {item.id}")

# %% [markdown]
# ### Los helpers de `utils.stac` para la búsqueda
#
# Lo que acabamos de escribir a mano se puede preparar con dos helpers, sin esconder la llamada nativa:
#
# * `parametros_busqueda(...)` devuelve el diccionario que se pasa a `catalogo.search(**parametros)` (colección, bbox, fechas y filtro de nubes).
# * `preparar_items(...)` limpia los resultados: una misma toma de Sentinel-2 puede aparecer varias veces (reprocesamientos de la ESA con distinta versión de procesamiento, o *baseline*), y deja una sola versión de cada una. Además exige que todas las imágenes vengan del mismo catálogo.

# %%
parametros = parametros_busqueda("sentinel-2", bbox, (fecha_inicio, fecha_fin), nubes_max=20, catalogo=catalog)
print(parametros)

# La búsqueda sigue siendo la llamada nativa de pystac_client
search = catalog.search(**parametros)
items = preparar_items(search.item_collection())

# %% [markdown]
# ## 3. Acceso y Visualización de Imágenes Sentinel-2
#
# Vamos a cargar y visualizar una imagen Sentinel-2. El satélite Sentinel-2 tiene 13 bandas espectrales:
#
# | Banda | Longitud de onda (nm) | Resolución (m) | Descripción |
# |-------|----------------------|----------------|-------------|
# | B01   | 443                  | 60            | Aerosoles   |
# | B02   | 490                  | 10            | Azul        |
# | B03   | 560                  | 10            | Verde       |
# | B04   | 665                  | 10            | Rojo        |
# | B05   | 705                  | 20            | Red Edge 1  |
# | B06   | 740                  | 20            | Red Edge 2  |
# | B07   | 783                  | 20            | Red Edge 3  |
# | B08   | 842                  | 10            | NIR         |
# | B8A   | 865                  | 20            | Red Edge 4  |
# | B09   | 940                  | 60            | Vapor agua  |
# | B10   | 1375                 | 60            | Cirrus      |
# | B11   | 1610                 | 20            | SWIR 1      |
# | B12   | 2190                 | 20            | SWIR 2      |

# %%
# Cargamos las bandas RGB (color verdadero)
rgb_bands = ["B04", "B03", "B02"]  # Rojo, Verde, Azul

# Cargamos los datos usando ODC
ds = odc.stac.load(
    items,
    bands=rgb_bands,
    bbox=bbox,
    crs="EPSG:32719",  # UTM Zone 19S
    resolution=10,  # 10m resolución
    groupby="solar_day",  # Agrupamos por día solar: imágenes del mismo día se combinan en una fecha
    chunks={"x": 2048, "y": 2048},  # Tamaño de los chunks
)
ds

# %% [markdown]
# Los datos vienen como **números digitales (DN)**, no como reflectancia. Para convertirlos usamos `convertir_a_reflectancia`, que aplica `reflectancia = DN * factor + offset`:
#
# * Sentinel-2: `DN * 0.0001 + offset`. El offset **no es el mismo para todas las imágenes**: es -0.1 en las imágenes con versión de procesamiento (*baseline*) 04.00 o posterior, introducida el 25 de enero de 2022, y 0 en las anteriores. Qué imágenes lo llevan depende del catálogo y del baseline de cada una, no solo de la fecha (por ejemplo, hay tomas de 2021 reprocesadas con baseline 04.00). Sin el offset correcto, la reflectancia queda sobreestimada o subestimada en cerca de 0.1.
# * Landsat: el factor y el offset los informa el propio catálogo (lo veremos más abajo).
#
# Con la llamada de arriba no sabemos el offset de cada fecha, así que usamos `parametros_carga(...)`: devuelve los argumentos de `odc.stac.load` (bandas, CRS, resolución, `chunks`, firma de los enlaces y un `groupby` que **nunca junta en una misma fecha imágenes con offsets distintos**). Después `agregar_offset(datos, items)` guarda el offset de cada fecha en la coordenada `offset_reflectancia`, y `convertir_a_reflectancia` la usa. La carga sigue siendo la llamada nativa `odc.stac.load`.

# %%
parametros_ds = parametros_carga(
    "sentinel-2",
    rgb_bands,
    resolucion=10,
    crs="EPSG:32719",
    catalogo=catalog,
    bbox=bbox,
    chunks={"x": 2048, "y": 2048},
)
print(parametros_ds)

ds = odc.stac.load(items, **parametros_ds)
ds = agregar_offset(ds, items)
ds

# %% [markdown]
# Luego `crear_imagen_rgb` apila las tres bandas en un arreglo (alto, ancho, 3), multiplica por un factor de brillo (3.5, ajustable) y recorta los valores entre 0 y 1.

# %%
# Seleccionamos una fecha específica
fecha = ds.time[0]
imagen = ds.sel(time=fecha)

# DN -> reflectancia, y luego las tres bandas apiladas como imagen RGB
reflectancia = convertir_a_reflectancia(imagen, "sentinel-2")
rgb = crear_imagen_rgb(reflectancia, rgb_bands, factor_brillo=3.5)

mostrar_imagen(rgb, f"Imagen Sentinel-2 RGB - Color verdadero ({str(fecha.values)[:10]})")

# %%
# Visualización en falso color (NIR)
nir_bands = ["B08", "B04", "B03"]  # NIR, Rojo, Verde

# Cargamos las otras bandas con la misma receta: parametros_carga + odc.stac.load + agregar_offset
ds_nir = odc.stac.load(
    items,
    **parametros_carga("sentinel-2", nir_bands, resolucion=10, crs="EPSG:32719", catalogo=catalog,
                       bbox=bbox, chunks={"x": 2048, "y": 2048}),
)
ds_nir = agregar_offset(ds_nir, items)

# Seleccionamos la misma fecha
imagen_nir = ds_nir.sel(time=fecha)

# Visualizamos la imagen en falso color
nir_rgb = crear_imagen_rgb(convertir_a_reflectancia(imagen_nir, "sentinel-2"), nir_bands, factor_brillo=3.5)

mostrar_imagen(nir_rgb, f"Imagen Sentinel-2 - Falso color NIR ({str(fecha.values)[:10]})")

# %% [markdown]
# ## 4. Acceso a Imágenes Landsat
#
# Ahora vamos a buscar y visualizar imágenes Landsat para la misma región.

# %%
# Búsqueda de imágenes Landsat (misma área y período), reutilizando la conexión al catálogo
parametros_ls = parametros_busqueda("landsat", bbox, (fecha_inicio, fecha_fin), nubes_max=20, catalogo=catalog)
items_landsat = preparar_items(catalog.search(**parametros_ls).item_collection())

if len(items_landsat) > 0:
    # Mostramos información de la primera imagen
    item_landsat = items_landsat[0]
    print("\nInformación de la primera imagen Landsat:")
    print(f"Fecha: {item_landsat.datetime.strftime('%Y-%m-%d')}")
    print(f"Cobertura de nubes: {item_landsat.properties['eo:cloud_cover']}%")
    print(f"ID de la imagen: {item_landsat.id}")

    # Cargamos las bandas RGB de Landsat
    rgb_bands_landsat = ["red", "green", "blue"]  # Rojo, Verde, Azul
    ds_landsat = odc.stac.load(
        items_landsat,
        **parametros_carga("landsat", rgb_bands_landsat, resolucion=30,  # 30 m resolución para Landsat
                           crs="EPSG:32719", catalogo=catalog, bbox=bbox, chunks={"x": 2048, "y": 2048}),
    )

    # Seleccionamos una fecha
    fecha_landsat = ds_landsat.time[0]
    imagen_landsat = ds_landsat.sel(time=fecha_landsat)

# %% [markdown]
# En Landsat el factor de escala y el offset **no son fijos**: vienen en los metadatos STAC de cada banda (`raster:bands`). La función `escala_y_offset` los lee por nosotros.

# %%
if len(items_landsat) > 0:
    # Leemos el factor de escala y el offset de la banda "blue" desde los metadatos
    escala, offset = escala_y_offset(items_landsat[0], "blue")
    print(f"Landsat: reflectancia = DN * {escala} + ({offset})")

    # Convertimos a reflectancia "a mano" (convertir_a_reflectancia hace exactamente esto)
    reflectancia_landsat = imagen_landsat * escala + offset
    rgb_landsat = crear_imagen_rgb(reflectancia_landsat, rgb_bands_landsat, factor_brillo=3.5)

    mostrar_imagen(rgb_landsat,
                   f"Imagen Landsat RGB - Color verdadero ({str(fecha_landsat.values)[:10]})")

# %% [markdown]
# ## 5. Comparación entre Sentinel-2 y Landsat
#
# Principales diferencias:
#
# * **Resolución espacial**:
#   * Sentinel-2: 10m en bandas visibles y NIR
#   * Landsat 8: 30m en la mayoría de las bandas
# * **Resolución temporal**:
#   * Sentinel-2: 5 días (con dos satélites)
#   * Landsat 8: 16 días
# * **Bandas espectrales**:
#   * Sentinel-2: 13 bandas
#   * Landsat 8: 11 bandas
# * **Cobertura**:
#   * Sentinel-2: Optimizado para tierra
#   * Landsat 8: Global, incluye océanos

# %% [markdown]
# ## Ejercicios
#
# 1. Busca imágenes Sentinel-2 para otra región de Chile y visualízalas en color verdadero.
# 2. Experimenta con diferentes combinaciones de bandas para crear visualizaciones en falso color.
# 3. Compara la misma área en imágenes Sentinel-2 y Landsat. ¿Puedes notar las diferencias en resolución?
# 4. Modifica los parámetros de búsqueda (fechas, cobertura de nubes) y analiza cómo afectan los resultados.
