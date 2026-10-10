# %% [markdown]
# # Ejercicio: Histogramas Landsat Interactivo
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/05_ejercicio_histograma_landsat.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# Descarga una imagen Landsat, calcula histogramas de todas las bandas y selecciona una región interactivamente.
#
# **Pasos:**
# 1. Buscar y cargar una imagen Landsat de la zona central de Chile
# 2. Convertirla a reflectancia y verla en color verdadero
# 3. Ver el histograma de todas las bandas de la escena
# 4. Dibujar un rectángulo en un mapa y comparar el histograma de esa región
# 5. Exportar la escena como GeoTIFF

# %% [markdown]
# ## Configuración inicial
#
# La celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (buscar imágenes, convertir a reflectancia, histogramas, selector de región, etc.).

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (solo la primera vez)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalación de paquetes
# %pip install -q rioxarray xarray matplotlib numpy planetary-computer pystac-client odc-stac ipyleaflet plotly pyproj ipython

# %%
# @title Imports
import odc.stac
import rioxarray  # noqa: F401  (activa .rio para exportar GeoTIFF)
import xarray as xr

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import convertir_a_reflectancia, nombres_bandas, recortar_region
from utils.stac import conectar_catalogo, escala_y_offset, parametros_busqueda, parametros_carga, preparar_items
from utils.viz import SelectorRegion, crear_imagen_rgb, histograma_interactivo, mostrar_imagen

# %% [markdown]
# ## 1. Cargar la imagen Landsat
#
# Buscamos imágenes Landsat 8/9 con poca nubosidad en el Valle Central (enero de 2025) y cargamos todas las bandas ópticas. Los datos llegan como números digitales (DN); los convertimos a reflectancia con `convertir_a_reflectancia`.

# %%
# @title Cargar Landsat
# Área de interés: Valle Central, Chile
bbox = [-70.8, -33.8, -70.6, -33.6]  # [oeste, sur, este, norte]

# Buscar Landsat 8/9 con menos de 20% de nubes (búsqueda nativa; los helpers arman los parámetros)
catalogo = conectar_catalogo("planetary")
parametros = parametros_busqueda("landsat", bbox, ("2025-01-01", "2025-01-31"), nubes_max=20, catalogo=catalogo)
items = preparar_items(catalogo.search(**parametros).item_collection())

# Cargar todas las bandas ópticas (nombres genéricos -> nombres Landsat)
bandas = nombres_bandas("landsat", ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"])
print(f"Bandas: {bandas}")

# Carga nativa con odc.stac; `parametros_carga` arma los argumentos
ds = odc.stac.load(items, **parametros_carga("landsat", bandas, resolucion=30, crs="EPSG:32719", bbox=bbox))

# Aplicar scale/offset para reflectancia (los valores vienen en los metadatos de cada imagen)
escala, offset = escala_y_offset(items[0], "red")
print(f"Reflectancia = DN * {escala} + ({offset})")

ds_refl = convertir_a_reflectancia(ds, "landsat")
imagen = ds_refl.isel(time=0)

print(f"Dataset cargado: {dict(imagen.sizes)}")

# %% [markdown]
# ## 2. Visualizar en color verdadero

# %%
# @title Visualizar RGB
rgb_img = crear_imagen_rgb(imagen, ["red", "green", "blue"], factor_brillo=3.0)

mostrar_imagen(rgb_img, f"Imagen Landsat - {str(imagen.time.values)[:10]}", figsize=(12, 10))

# %% [markdown]
# ## 3. Histograma de la escena completa
#
# Cada línea es el histograma de una banda (los colores recuerdan la región del espectro: azul, verde, rojo, infrarrojo...).

# %%
# @title Histograma Completo (Plotly)
histograma_interactivo(imagen, bandas, "Histogramas - Escena Completa")

# %% [markdown]
# ## 4. Selección interactiva de una región
#
# En el mapa siguiente, usa la herramienta de rectángulo (a la izquierda) para dibujar una región sobre la imagen. Las coordenadas quedan guardadas en `selector.limites`.

# %%
# @title Selección Interactiva (ipyleaflet)
selector = SelectorRegion(rgb_img, bbox)
selector.mostrar()

# %% [markdown]
# Cuando hayas dibujado el rectángulo, ejecuta la celda siguiente para ver el histograma de esa región.

# %%
# @title Histograma Región Seleccionada
if selector.limites is not None:
    # El rectángulo viene en grados (EPSG:4326) y la imagen está en UTM 19S (EPSG:32719):
    # recortar_region hace la transformación de coordenadas y el recorte.
    region = recortar_region(imagen, selector.limites)
    print(f"Región recortada: {dict(region.sizes)}")

    # Histograma de la región
    histograma_interactivo(region, bandas, "Histogramas - Región Seleccionada")

    # Mostrar la región recortada
    rgb_region = crear_imagen_rgb(region, ["red", "green", "blue"], factor_brillo=3.0)
    mostrar_imagen(rgb_region, "Región Seleccionada - Color Verdadero", figsize=(10, 8))
else:
    print("Primero selecciona una región dibujando un rectángulo en el mapa")

# %% [markdown]
# ## 5. Exportar como GeoTIFF
#
# Apilamos todas las bandas en un solo arreglo y lo guardamos como GeoTIFF multibanda, que puedes abrir en QGIS u otro software SIG.

# %%
# @title Export GeoTIFF
# Combinar todas las bandas en un solo DataArray
banda_stack = xr.concat([imagen[b] for b in bandas], dim="band")
banda_stack = banda_stack.assign_coords(band=bandas)

# Exportar como GeoTIFF multi-banda
banda_stack.rio.to_raster("landsat_scene.tif", driver="GTiff")

# Mostrar información del archivo
print("\nInformación del archivo:")
print(f"  Dimensiones: {banda_stack.sizes}")
print(f"  Bandas: {list(banda_stack.band.values)}")
print(f"  CRS: {banda_stack.rio.crs}")
print(f"  Resolución: {banda_stack.rio.resolution()}")
