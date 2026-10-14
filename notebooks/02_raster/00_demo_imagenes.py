# %% [markdown]
# # Demo: Exploración de Imágenes Satelitales con STAC
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/00_demo_imagenes.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# En este cuaderno demostraremos las capacidades de un catálogo STAC y las funcionalidades
# básicas para trabajar con imágenes satelitales. Aprenderemos a:
#
# * Buscar y acceder imágenes Sentinel-2 y Landsat
# * Visualizar bandas individuales y combinaciones
# * Comparar números digitales vs. reflectancia
# * Calcular índices espectrales
# * Explorar las imágenes y extraer series temporales de puntos

# %% [markdown]
# ## Configuración del entorno
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (conectar al catálogo, buscar y cargar imágenes, convertir a reflectancia, mostrar imágenes, series temporales, etc.). Las iremos conociendo a lo largo del cuaderno.

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
# %pip install -q rioxarray xarray matplotlib numpy geopandas planetary-computer pystac-client odc-stac ipywidgets plotly

# %%
# @title Importación de bibliotecas
import odc.stac

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import calcular_indices, convertir_a_reflectancia, nombres_bandas
from utils.stac import (
    agregar_offset,
    conectar_catalogo,
    escala_y_offset,
    parametros_busqueda,
    parametros_carga,
    preparar_items,
    renombrar_bandas,
    resumir_imagenes,
)
from utils.viz import (
    crear_imagen_rgb,
    explorador_imagenes,
    graficar_series_puntos,
    mostrar_imagen,
    mostrar_paneles,
)

# %% [markdown]
# ## 1. Definición del Área de Interés
#
# **Tip**: Para encontrar las coordenadas de tu área de interés, puedes usar [bboxfinder.com](http://bboxfinder.com/)
#
# 1. Ve a bboxfinder.com
# 2. Navega hasta tu área de interés
# 3. Dibuja un rectángulo sobre la zona
# 4. Copia las coordenadas en formato [minx, miny, maxx, maxy]
#
# Para este demo usaremos un área cerca de Santiago, Chile:

# %%
# Área de interés: Región Metropolitana, Chile
# Formato: [longitud_oeste, latitud_sur, longitud_este, latitud_norte]
bbox = [-71.399460, -34.366111, -70.633850, -34.084512]

print(f"Bounding box: {bbox}")
print("Coordenadas:")
print(f"  Longitud: {bbox[0]:.6f} a {bbox[2]:.6f}")
print(f"  Latitud: {bbox[1]:.6f} a {bbox[3]:.6f}")

# %% [markdown]
# ## 2. Conexión al Catálogo STAC
#
# STAC (SpatioTemporal Asset Catalog) es un estándar para organizar y descubrir
# datos geoespaciales. Planetary Computer ofrece acceso gratuito a múltiples
# colecciones de datos satelitales.
#
# La función `conectar_catalogo()` abre el catálogo con `pystac_client.Client.open(...)` y devuelve
# el cliente nativo (en Planetary Computer firma automáticamente los enlaces de descarga).
#
# > **Cómo usamos `utils.stac`:** las funciones de ese módulo solo **preparan** las llamadas nativas.
# > Tú ejecutas `catalog.search(...)` y `odc.stac.load(...)`; los helpers arman sus argumentos y
# > ordenan los datos antes y después.

# %%
# Conectamos al catálogo STAC de Planetary Computer (cliente nativo de pystac_client)
catalog = conectar_catalogo("planetary")

print("Conectado exitosamente al catálogo STAC de Planetary Computer")

# %% [markdown]
# ## 3. Búsqueda de Imágenes Sentinel-2
#
# ### Bandas Sentinel-2 (MSI - MultiSpectral Instrument)
#
# | Banda | Nombre | Rango espectral (nm) | Resolución (m) | Uso principal |
# |-------|--------|---------------------|----------------|---------------|
# | B01   | Aerosol | 433-453 | 60 | Corrección atmosférica |
# | B02   | Azul | 458-523 | 10 | Masas de agua, atmósfera |
# | B03   | Verde | 543-578 | 10 | Vegetación saludable |
# | B04   | Rojo | 650-680 | 10 | Vegetación, suelos |
# | B05   | Red Edge 1 | 698-713 | 20 | Estado de vegetación |
# | B06   | Red Edge 2 | 733-748 | 20 | Estado de vegetación |
# | B07   | Red Edge 3 | 773-793 | 20 | Estado de vegetación |
# | B08   | NIR | 785-900 | 10 | Biomasa, líneas costeras |
# | B8A   | Red Edge 4 | 855-875 | 20 | Estado de vegetación |
# | B09   | Vapor agua | 935-955 | 60 | Corrección atmosférica |
# | B11   | SWIR 1 | 1565-1655 | 20 | Humedad vegetación/suelo |
# | B12   | SWIR 2 | 2100-2280 | 20 | Geología, humedad |
#
# **Para usar Landsat en su lugar, cambia `"sentinel-2"` por `"landsat"` en `parametros_busqueda` (más abajo lo hacemos)**
#
# ### Bandas Landsat 8/9 (OLI - Operational Land Imager)
#
# | Banda | Nombre | Rango espectral (nm) | Resolución (m) | Uso principal |
# |-------|--------|---------------------|----------------|---------------|
# | B1    | Aerosol | 435-451 | 30 | Corrección atmosférica |
# | B2    | Azul | 452-512 | 30 | Masas de agua, atmósfera |
# | B3    | Verde | 533-590 | 30 | Vegetación saludable |
# | B4    | Rojo | 636-673 | 30 | Vegetación, suelos |
# | B5    | NIR | 851-879 | 30 | Biomasa, líneas costeras |
# | B6    | SWIR 1 | 1566-1651 | 30 | Humedad vegetación/suelo |
# | B7    | SWIR 2 | 2107-2294 | 30 | Geología, humedad |


# %%
# Período de tiempo para la búsqueda
fecha_inicio, fecha_fin = "2023-08-01", "2023-08-31"  # Agosto 2023 (estación seca)

# `parametros_busqueda` arma el diccionario de la búsqueda (colección, bbox, fechas y filtro de nubes):
# sensor="sentinel-2" (o "landsat"), menos de 15% de nubes
parametros = parametros_busqueda(
    "sentinel-2", bbox, (fecha_inicio, fecha_fin), nubes_max=15, catalogo=catalog
)
print(parametros)

# La búsqueda es la llamada nativa de pystac_client
busqueda = catalog.search(**parametros)
items = busqueda.item_collection()

# `preparar_items` quita las versiones repetidas de una misma toma y ordena por fecha
items = preparar_items(items)

# Resumen de las imágenes encontradas (fecha, % de nubes, identificador, baseline y offset)
resumir_imagenes(items)

# %% [markdown]
# ## 4. Carga de Bandas Seleccionadas
#
# Cargaremos varias bandas clave para análisis multispectral:
# * B02 (Azul), B03 (Verde), B04 (Rojo): Para color verdadero
# * B05 (Red Edge), B08 (NIR): Para análisis de vegetación
# * B11 (SWIR 1), B12 (SWIR 2): Para análisis de humedad y geología

# %%
# Selección de bandas clave
bandas_seleccionadas = ["B02", "B03", "B04", "B05", "B08", "B11", "B12"]

# `parametros_carga` arma los argumentos de odc.stac.load: bandas, CRS, resolución, chunks, firma de los
# enlaces y cómo agrupar las imágenes por fecha
parametros_ds = parametros_carga(
    "sentinel-2",
    bandas_seleccionadas,
    resolucion=20,  # 20m resolución (compromiso entre calidad y velocidad)
    crs="EPSG:32719",  # UTM Zone 19S para Chile central
    catalogo=catalog,
    bbox=bbox,
    chunks={"x": 1024, "y": 1024},  # de forma perezosa: se descargan solo cuando los usamos
)

# La carga es la llamada nativa de odc.stac
ds = odc.stac.load(items, **parametros_ds)

# Dejamos los nombres de banda estándar (B02, B04...) y agregamos el offset de reflectancia de cada fecha
ds = renombrar_bandas(ds, catalogo=catalog)
ds = agregar_offset(ds, items)

print("Dataset cargado:")
print(f"  Dimensiones: {dict(ds.sizes)}")
print(f"  Bandas: {list(ds.data_vars)}")
print(f"  Fechas: {len(ds.time)} imágenes")

# Seleccionamos una fecha y la descargamos a memoria (.compute())
fecha_seleccionada = ds.time[1]
imagen = ds.sel(time=fecha_seleccionada).compute()
print(f"\nTrabajando con imagen del: {str(fecha_seleccionada.values)[:10]}")

# %% [markdown]
# ## 5. Visualización de Bandas Individuales
#
# Antes de crear composiciones, veamos cómo se ven las bandas individuales en escala de grises:

# %%
# Visualizamos bandas individuales en escala de grises (números digitales)
opciones_gris = {"cmap": "gray", "vmin": 0, "vmax": 5000, "barra": False}

mostrar_paneles(
    [
        ("Banda Verde (B03)\n560 nm - Vegetación saludable", imagen.B03, opciones_gris),
        ("Banda Roja (B04)\n665 nm - Vegetación y suelos", imagen.B04, opciones_gris),
        ("Banda NIR (B08)\n842 nm - Biomasa vegetal", imagen.B08, opciones_gris),
    ],
    ncols=3,
    titulo="Bandas Individuales - Números Digitales",
)

print(
    "Nota: Los valores más altos (más brillantes) indican mayor reflectancia en esa longitud de onda"
)

# %% [markdown]
# ## 6. Números Digitales vs. Reflectancia
#
# Los sensores guardan los datos como números digitales (DN). Para análisis científico,
# debemos convertirlos a reflectancia usando factores de escala. Para Sentinel-2 L2A:
#
# `reflectancia = DN * 0.0001 + offset`
#
# El factor 0.0001 es el mismo para todas las imágenes, pero el offset no: es -0.1 en las imágenes con
# versión de procesamiento (baseline) 04.00 o posterior, que la ESA introdujo el 25 de enero de 2022 y que
# guardan los DN con un desfase de +1000, y 0 en las anteriores. Qué imágenes lo llevan depende del catálogo
# y del baseline de cada una (hay tomas de 2021 reprocesadas con baseline 04.00), no solo de la fecha.
# Si lo olvidamos, la reflectancia queda sobreestimada en cerca de 0.1.
#
# Los helpers lo resuelven imagen por imagen: `parametros_carga` evita mezclar offsets distintos en una
# misma fecha y `agregar_offset` guarda el offset de cada fecha en la coordenada `offset_reflectancia`.
# La función `convertir_a_reflectancia` la usa para aplicar la fórmula (y descarta los valores fuera del
# rango 0 a 1).

# %%
# Conversión a reflectancia
imagen_reflectancia = convertir_a_reflectancia(imagen, "sentinel-2")

print("Comparación de rangos de valores:")
print("Números digitales (DN):")
print(f"  Verde: {imagen.B03.min().values:.0f} - {imagen.B03.max().values:.0f}")
print(f"  Rojo:  {imagen.B04.min().values:.0f} - {imagen.B04.max().values:.0f}")
print(f"  NIR:   {imagen.B08.min().values:.0f} - {imagen.B08.max().values:.0f}")

print("\nReflectancia (0-1):")
print(
    f"  Verde: {imagen_reflectancia.B03.min().values:.3f} - {imagen_reflectancia.B03.max().values:.3f}"
)
print(
    f"  Rojo:  {imagen_reflectancia.B04.min().values:.3f} - {imagen_reflectancia.B04.max().values:.3f}"
)
print(
    f"  NIR:   {imagen_reflectancia.B08.min().values:.3f} - {imagen_reflectancia.B08.max().values:.3f}"
)

# %% [markdown]
# ## 7. Visualización RGB (Color Verdadero)
#
# `crear_imagen_rgb` apila tres bandas (en el orden rojo, verde, azul de la pantalla), multiplica por un
# factor de brillo y recorta los valores entre 0 y 1, dejando la imagen lista para mostrar.

# %%
# Composición RGB - Color verdadero
rgb_verdadero = crear_imagen_rgb(imagen_reflectancia, ["B04", "B03", "B02"], factor_brillo=3.5)

mostrar_imagen(
    rgb_verdadero,
    f"Imagen Sentinel-2 - Color Verdadero\nFecha: {str(fecha_seleccionada.values)[:10]}",
)

# %% [markdown]
# ## 8. Combinaciones en Falso Color
#
# Las combinaciones en falso color resaltan diferentes características del paisaje:

# %% [markdown]
# ### Falso Color Infrarrojo (NIR-R-G)
# **Uso**: Análisis de vegetación - La vegetación aparece en tonos rojos brillantes

# %%
# Falso color infrarrojo: NIR, Rojo, Verde
falso_color_nir = crear_imagen_rgb(imagen_reflectancia, ["B08", "B04", "B03"])

mostrar_imagen(falso_color_nir, "Falso Color Infrarrojo (NIR-R-G)\nVegetación = Rojo brillante")

# %% [markdown]
# ### Combinación Agricultura (SWIR1-NIR-R)
# **Uso**: Monitoreo agrícola y análisis de cultivos - Resalta diferentes tipos de vegetación y humedad

# %%
# Combinación agricultura: SWIR1, NIR, Rojo
agricultura = crear_imagen_rgb(imagen_reflectancia, ["B11", "B08", "B04"], factor_brillo=4.0)

mostrar_imagen(agricultura, "Combinación Agricultura (SWIR1-NIR-R)\nCultivos y humedad del suelo")

# %% [markdown]
# ### Combinación Geología (SWIR2-SWIR1-NIR)
# **Uso**: Análisis geológico - Resalta minerales y rocas, útil para prospección minera
#
# **Tip**: Puedes experimentar con diferentes combinaciones modificando las bandas:
# * **Urbano**: SWIR2-NIR-R (B12-B08-B04) - Distingue áreas urbanas
# * **Agua**: NIR-SWIR1-R (B08-B11-B04) - Resalta cuerpos de agua
# * **Nieve**: R-G-SWIR1 (B04-B03-B11) - Distingue nieve de nubes

# %%
# Combinación geología: SWIR2, SWIR1, NIR
# Opciones disponibles: B02, B03, B04, B05, B08, B11, B12

banda_roja = "B12"  # SWIR2 para geología
banda_verde = "B11"  # SWIR1
banda_azul = "B08"  # NIR

# Factor de brillo para mejor visualización
factor_brillo_geologia = 4.0  # Ajusta entre 2.0 y 6.0 según necesites

# Crear la composición
composicion_geologia = crear_imagen_rgb(
    imagen_reflectancia, [banda_roja, banda_verde, banda_azul], factor_brillo_geologia
)

mostrar_imagen(
    composicion_geologia,
    f"Combinación Geología: {banda_roja}-{banda_verde}-{banda_azul}\nFactor de brillo: {factor_brillo_geologia}",
)

print(f"Combinación actual: R={banda_roja}, G={banda_verde}, B={banda_azul}")

# %% [markdown]
# ## 9. Ejemplo con Landsat (4 meses de datos)
#
# Ahora trabajaremos con Landsat para tener más datos temporales:

# %%
# Búsqueda de imágenes Landsat para 4 meses (enero a abril 2024): mismo flujo, mismo catálogo
parametros_ls = parametros_busqueda(
    "landsat",
    bbox,
    ("2024-01-01", "2024-04-30"),
    nubes_max=30,  # Menos restrictivo para tener más imágenes
    catalogo=catalog,
)
items_landsat = preparar_items(catalog.search(**parametros_ls).item_collection())

if len(items_landsat) > 0:
    # Cargamos bandas equivalentes de Landsat (nombres genéricos -> nombres Landsat)
    bandas_landsat = nombres_bandas("landsat", ["red", "green", "blue", "nir", "swir1", "swir2"])
    print(f"Bandas Landsat: {bandas_landsat}")

    ds_landsat = odc.stac.load(
        items_landsat,
        **parametros_carga(
            "landsat",
            bandas_landsat,
            resolucion=30,  # Resolución nativa de Landsat
            crs="EPSG:32719",
            catalogo=catalog,
            bbox=bbox,
        ),
    )

    print("\nDataset Landsat cargado:")
    print(f"  Dimensiones: {dict(ds_landsat.sizes)}")
    print(f"  Fechas disponibles: {len(ds_landsat.time)}")

    # En Landsat, el factor de escala y el offset vienen en los metadatos STAC de cada imagen
    scale, offset = escala_y_offset(items_landsat[0], "red")

    print(f"  Factor de escala: {scale}")
    print(f"  Offset: {offset}")

    # Seleccionar una imagen del medio del período
    fecha_landsat = ds_landsat.time[len(ds_landsat.time) // 2]
    imagen_landsat = ds_landsat.sel(time=fecha_landsat)

    # Convertir a reflectancia: DN * escala + offset
    imagen_landsat_refl = imagen_landsat * scale + offset

    print(f"\nTrabajando con imagen Landsat del: {str(fecha_landsat.values)[:10]}")

# %% [markdown]
# ### Visualización RGB Landsat

# %%
if len(items_landsat) > 0:
    # RGB Landsat: rojo, verde, azul
    rgb_landsat = crear_imagen_rgb(imagen_landsat_refl, ["red", "green", "blue"], factor_brillo=3.0)

    mostrar_imagen(
        rgb_landsat, f"Landsat 8/9 - Color Verdadero\nFecha: {str(fecha_landsat.values)[:10]}"
    )

# %% [markdown]
# ### Falso Color Landsat

# %%
if len(items_landsat) > 0:
    # Falso color NIR-R-G para Landsat: nir08 como rojo, red como verde, green como azul
    falso_landsat = crear_imagen_rgb(
        imagen_landsat_refl, ["nir08", "red", "green"], factor_brillo=3.5
    )

    mostrar_imagen(
        falso_landsat,
        f"Landsat 8/9 - Falso Color Infrarrojo\nFecha: {str(fecha_landsat.values)[:10]}",
    )

# %% [markdown]
# ## 10. Cálculo de Índices Espectrales
#
# Antes de calcular los índices espectrales, hagamos un pequeño experimento, mencionado en clases:
# ¿qué pasa si no convertimos los números digitales a reflectancia antes de calcular los índices espectrales?
#
# Recuerda la fórmula del **NDVI**: `(NIR - ROJO) / (NIR + ROJO)`.

# %%
ndvi_dn = (imagen_landsat.nir08 - imagen_landsat.red) / (
    imagen_landsat.nir08 + imagen_landsat.red
)
ndvi_refl = (imagen_landsat_refl.nir08 - imagen_landsat_refl.red) / (
    imagen_landsat_refl.nir08 + imagen_landsat_refl.red
)

# Visualización comparativa
opciones_ndvi = {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.8}

mostrar_paneles(
    [
        ("NDVI - Números Digitales", ndvi_dn, opciones_ndvi),
        ("NDVI - Reflectancia", ndvi_refl, opciones_ndvi),
    ],
    ncols=2,
    titulo="Comparación NDVI: Números Digitales vs Reflectancia",
)

# Estadísticas comparativas
print("Estadísticas de NDVI:")
print(
    f"Con números digitales: {ndvi_dn.mean().values:.4f} ± {ndvi_dn.std().values:.4f}"
)
print(
    f"Con reflectancia:      {ndvi_refl.mean().values:.4f} ± {ndvi_refl.std().values:.4f}"
)
print(f"Diferencia máxima:     {abs(ndvi_refl - ndvi_dn).max().values:.6f}")

# %% [markdown]
# Ahora sí, calculamos NDVI y EVI. Primero "a mano", para una sola fecha (así se ve la fórmula completa):
#
# * **NDVI** = (NIR - ROJO) / (NIR + ROJO)
# * **EVI** = 2.5 * (NIR - ROJO) / (NIR + 6 * ROJO - 7.5 * AZUL + 1)

# %%
if len(items_landsat) > 0:
    nir = imagen_landsat_refl.nir08
    rojo = imagen_landsat_refl.red
    azul = imagen_landsat_refl.blue

    ndvi_manual = (nir - rojo) / (nir + rojo)
    evi_manual = 2.5 * (nir - rojo) / (nir + 6 * rojo - 7.5 * azul + 1)

    print(f"NDVI de la fecha seleccionada: {float(ndvi_manual.mean()):.3f} (promedio)")
    print(f"EVI de la fecha seleccionada:  {float(evi_manual.mean()):.3f} (promedio)")

# %% [markdown]
# Para todo el stack temporal usamos `calcular_indices`, que aplica las mismas fórmulas a todas las fechas
# a la vez (y descarta los valores fuera del rango -1 a 1):

# %%
if len(items_landsat) > 0:
    # DN -> reflectancia para todas las fechas (mismo factor y offset que arriba)
    ds_landsat_refl = convertir_a_reflectancia(ds_landsat, "landsat")

    # Calcular NDVI y EVI para todo el stack temporal
    indices = calcular_indices(ds_landsat_refl, "landsat", ("NDVI", "EVI"))

    # Agregar al dataset
    ds_landsat_refl = ds_landsat_refl.assign(NDVI=indices.NDVI, EVI=indices.EVI)

    print(f"Índices calculados para {len(ds_landsat_refl.time)} fechas")
    print(f"Variables en el dataset: {list(ds_landsat_refl.data_vars)}")

    # Visualizar EVI de la fecha seleccionada
    evi_fecha = ds_landsat_refl.EVI.sel(time=fecha_landsat)

    mostrar_imagen(
        evi_fecha,
        f"Enhanced Vegetation Index (EVI)\nFecha: {str(fecha_landsat.values)[:10]}",
        cmap="RdYlGn",
        vmin=0,
        vmax=0.8,
        barra="EVI",
    )

# %% [markdown]
# ¿Cuál es la diferencia entre ambos objetos?

# %%
print(ds_landsat)
print(ds_landsat_refl)

# %% [markdown]
# ## 11. Exploración interactiva y series temporales
#
# ### Explorador de imágenes
#
# Con `explorador_imagenes` elegimos la fecha y el tipo de imagen (color verdadero, falso color, NDVI o EVI)
# desde dos menús. En los modos NDVI y EVI, al pasar el mouse sobre la imagen aparecen las coordenadas
# `x`, `y` (UTM 19S) y el valor del índice: sirven para elegir los puntos de la siguiente sección.

# %%
explorador_imagenes(ds_landsat_refl)

# %% [markdown]
# ### Series temporales en puntos
#
# Una **serie temporal** muestra cómo cambia un valor (por ejemplo el NDVI) en un mismo píxel a lo largo
# del tiempo. `graficar_series_puntos` toma una lista de coordenadas `[x, y]` (en el mismo sistema de
# coordenadas del dataset, aquí UTM 19S en metros), busca el píxel más cercano a cada una y grafica la
# serie.

# %%
# Coordenadas de tres puntos de interés [x, y] en UTM 19S (puedes cambiarlas por otras de tu área)
my_coordinates = [[307965, 6217395], [323505, 6201135], [297585, 6223635]]

graficar_series_puntos(ds_landsat_refl, my_coordinates, variables=("NDVI",))

# %%
# Mismos puntos, con NDVI y EVI en gráficos separados
graficar_series_puntos(ds_landsat_refl, my_coordinates, variables=("NDVI", "EVI"))
