# %% [markdown]
# # Demo: Exploración de Imágenes Satelitales con STAC
#
# En este cuaderno demostraremos las capacidades de un catálogo STAC y las funcionalidades
# básicas para trabajar con imágenes satelitales. Aprenderemos a:
#
# * Buscar y acceder imágenes Sentinel-2 y Landsat
# * Visualizar bandas individuales y combinaciones
# * Comparar números digitales vs. reflectancia
# * Calcular índices espectrales
# * Extraer series temporales interactivamente

# %% [markdown]
# ## Configuración del entorno

# %%
# @title Instalación de paquetes necesarios
# %pip install rioxarray xarray matplotlib numpy geopandas planetary-computer pystac-client odc-stac ipywidgets plotly


# %%
# @title Importación de bibliotecas
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output, display
from pystac_client import Client

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

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

# %%
# Conectamos al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

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
# **Para usar Landsat en su lugar, cambia la colección por "landsat-c2-l2"**
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
time_range = "2023-08-01/2023-08-31"  # Agosto 2023 (estación seca)

# Realizamos la búsqueda
search = catalog.search(
    collections=["sentinel-2-l2a"],  # Cambiar por "landsat-c2-l2" para Landsat
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 15}},  # Menos de 15% de nubes
)

# Convertimos los resultados a lista
items = list(search.get_items())
print(f"Encontradas {len(items)} imágenes Sentinel-2")

if len(items) > 0:
    # Mostramos información de la primera imagen
    item = items[0]
    print("\nPrimera imagen disponible:")
    print(f"  Fecha: {item.datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cobertura de nubes: {item.properties['eo:cloud_cover']:.1f}%")
    print(f"  ID: {item.id}")
else:
    print(
        "No se encontraron imágenes. Intenta ampliar el rango de fechas o aumentar el umbral de nubes."
    )

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

# Cargamos los datos usando ODC
ds = odc.stac.load(
    items,
    bands=bandas_seleccionadas,
    bbox=bbox,
    crs="EPSG:32719",  # UTM Zone 19S para Chile central
    resolution=20,  # 20m resolución (compromiso entre calidad y velocidad)
    groupby="solar_day",
    chunks={"x": 1024, "y": 1024},
)

print("Dataset cargado:")
print(f"  Dimensiones: {dict(ds.dims)}")
print(f"  Bandas: {list(ds.data_vars)}")
print(f"  Fechas: {len(ds.time)} imágenes")

# Seleccionamos la primera fecha disponible
fecha_seleccionada = ds.time[1]
imagen = ds.sel(time=fecha_seleccionada)
print(f"\nTrabajando con imagen del: {fecha_seleccionada.values}")

# %% [markdown]
# ## 5. Visualización de Bandas Individuales
#
# Antes de crear composiciones, veamos cómo se ven las bandas individuales en escala de grises:

# %%
# Visualizamos bandas individuales en escala de grises
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Banda Verde (B03)
axes[0].imshow(imagen.B03, cmap="gray", vmin=0, vmax=5000)
axes[0].set_title("Banda Verde (B03)\n560 nm - Vegetación saludable")
axes[0].axis("off")

# Banda Roja (B04)
axes[1].imshow(imagen.B04, cmap="gray", vmin=0, vmax=5000)
axes[1].set_title("Banda Roja (B04)\n665 nm - Vegetación y suelos")
axes[1].axis("off")

# Banda NIR (B08)
axes[2].imshow(imagen.B08, cmap="gray", vmin=0, vmax=5000)
axes[2].set_title("Banda NIR (B08)\n842 nm - Biomasa vegetal")
axes[2].axis("off")

plt.tight_layout()
plt.suptitle("Bandas Individuales - Números Digitales", fontsize=16, y=1.02)
plt.show()

print(
    "Nota: Los valores más altos (más brillantes) indican mayor reflectancia en esa longitud de onda"
)

# %% [markdown]
# ## 6. Números Digitales vs. Reflectancia
#
# Los sensores guardan los datos como números digitales (DN). Para análisis científico,
# debemos convertirlos a reflectancia usando factores de escala:

# %%
# Factor de escala para Sentinel-2 L2A
factor_escala = 0.0001

# Conversión a reflectancia
imagen_reflectancia = imagen * factor_escala

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


# %%
# Composición RGB - Color verdadero
def crear_composicion_rgb(imagen_refl, r_band, g_band, b_band, factor_brillo=3.5):
    """
    Crea una composición RGB para visualización.

    Parámetros:
    - imagen_refl: Dataset con valores de reflectancia
    - r_band, g_band, b_band: nombres de las bandas para rojo, verde, azul
    - factor_brillo: factor para mejorar el contraste visual
    """
    # Extraer las bandas
    r = imagen_refl[r_band].values
    g = imagen_refl[g_band].values
    b = imagen_refl[b_band].values

    # Apilar y transponer para formato (alto, ancho, canales)
    rgb = np.stack([r, g, b], axis=-1)

    # Aplicar factor de brillo y recortar valores
    rgb_enhanced = np.clip(rgb * factor_brillo, 0, 1)

    return rgb_enhanced


# Crear composición RGB
rgb_verdadero = crear_composicion_rgb(imagen_reflectancia, "B04", "B03", "B02")

plt.figure(figsize=(15, 10))
plt.imshow(rgb_verdadero)
plt.title(
    f"Imagen Sentinel-2 - Color Verdadero\nFecha: {fecha_seleccionada.values}",
    fontsize=14,
)
plt.axis("off")
plt.show()

# %% [markdown]
# ## 8. Combinaciones en Falso Color
#
# Las combinaciones en falso color resaltan diferentes características del paisaje:

# %% [markdown]
# ### Falso Color Infrarrojo (NIR-R-G)
# **Uso**: Análisis de vegetación - La vegetación aparece en tonos rojos brillantes

# %%
# Falso color infrarrojo: NIR, Rojo, Verde
falso_color_nir = crear_composicion_rgb(imagen_reflectancia, "B08", "B04", "B03")

plt.figure(figsize=(15, 10))
plt.imshow(falso_color_nir)
plt.title("Falso Color Infrarrojo (NIR-R-G)\nVegetación = Rojo brillante", fontsize=14)
plt.axis("off")
plt.show()

# %% [markdown]
# ### Combinación Agricultura (SWIR1-NIR-R)
# **Uso**: Monitoreo agrícola y análisis de cultivos - Resalta diferentes tipos de vegetación y humedad

# %%
# Combinación agricultura: SWIR1, NIR, Rojo
agricultura = crear_composicion_rgb(
    imagen_reflectancia, "B11", "B08", "B04", factor_brillo=4.0
)

plt.figure(figsize=(15, 10))
plt.imshow(agricultura)
plt.title(
    "Combinación Agricultura (SWIR1-NIR-R)\nCultivos y humedad del suelo", fontsize=14
)
plt.axis("off")
plt.show()

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
composicion_geologia = crear_composicion_rgb(
    imagen_reflectancia, banda_roja, banda_verde, banda_azul, factor_brillo_geologia
)

plt.figure(figsize=(15, 10))
plt.imshow(composicion_geologia)
plt.title(
    f"Combinación Geología: {banda_roja}-{banda_verde}-{banda_azul}\nFactor de brillo: {factor_brillo_geologia}",
    fontsize=14,
)
plt.axis("off")
plt.show()

print(f"Combinación actual: R={banda_roja}, G={banda_verde}, B={banda_azul}")

# %% [markdown]
# ## 9. Ejemplo con Landsat (4 meses de datos)
#
# Ahora trabajaremos con Landsat para tener más datos temporales:

# %%
# Búsqueda de imágenes Landsat para 4 meses
tiempo_landsat = "2024-01-01/2024-04-30"  # Enero a Abril 2024

search_landsat = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime=tiempo_landsat,
    query={
        "eo:cloud_cover": {"lt": 30},  # Menos restrictivo para tener más imágenes
        "platform": {"in": ["landsat-8", "landsat-9"]},
    },
)

items_landsat = list(search_landsat.get_items())
print(f"Encontradas {len(items_landsat)} imágenes Landsat")

if len(items_landsat) > 0:
    # Cargamos bandas equivalentes de Landsat
    bandas_landsat = ["red", "green", "blue", "nir08", "swir16", "swir22"]

    ds_landsat = odc.stac.load(
        items_landsat,
        bands=bandas_landsat,
        bbox=bbox,
        crs="EPSG:32719",
        resolution=30,  # Resolución nativa de Landsat
        groupby="solar_day",
    )

    print("\nDataset Landsat cargado:")
    print(f"  Dimensiones: {dict(ds_landsat.dims)}")
    print(f"  Fechas disponibles: {len(ds_landsat.time)}")

    # Información de escalado para Landsat
    # Los datos de Landsat vienen pre-escalados, pero podemos verificar
    sample_item = items_landsat[0]
    scale_info = sample_item.assets["red"].to_dict()["raster:bands"][0]
    scale = scale_info["scale"]
    offset = scale_info["offset"]

    print(f"  Factor de escala: {scale}")
    print(f"  Offset: {offset}")

    # Seleccionar una imagen del medio del período
    fecha_landsat = ds_landsat.time[len(ds_landsat.time) // 2]
    imagen_landsat = ds_landsat.sel(time=fecha_landsat)

    # Convertir a reflectancia (Landsat ya viene escalado)
    imagen_landsat_refl = imagen_landsat * scale + offset

    print(f"\nTrabajando con imagen Landsat del: {fecha_landsat.values}")

# %% [markdown]
# ### Visualización RGB Landsat

# %%
if len(items_landsat) > 0:
    # RGB Landsat
    def crear_rgb_landsat(imagen, factor_brillo=3.0):
        r = imagen.red.values
        g = imagen.green.values
        b = imagen.blue.values
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * factor_brillo, 0, 1)

    rgb_landsat = crear_rgb_landsat(imagen_landsat_refl)

    plt.figure(figsize=(15, 10))
    plt.imshow(rgb_landsat)
    plt.title(
        f"Landsat 8/9 - Color Verdadero\nFecha: {fecha_landsat.values}", fontsize=14
    )
    plt.axis("off")
    plt.show()

# %% [markdown]
# ### Falso Color Landsat

# %%
if len(items_landsat) > 0:
    # Falso color NIR-R-G para Landsat
    def crear_falso_color_landsat(imagen, factor_brillo=3.5):
        r = imagen.nir08.values  # NIR como rojo
        g = imagen.red.values  # Rojo como verde
        b = imagen.green.values  # Verde como azul
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * factor_brillo, 0, 1)

    falso_landsat = crear_falso_color_landsat(imagen_landsat_refl)

    plt.figure(figsize=(15, 10))
    plt.imshow(falso_landsat)
    plt.title(
        f"Landsat 8/9 - Falso Color Infrarrojo\nFecha: {fecha_landsat.values}",
        fontsize=14,
    )
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## 10. Cálculo de Índices Espectrales
#
# Antes de calcular los índices espectrales, hagamos un pequeño experimento, mencionado en clases
# ¿Qué pasas si no convierte los números digitales a reflectancia antes de calcular los índices espectrales?

# %%
ndvi_dn = (imagen_landsat.nir08 - imagen_landsat.red) / (
    imagen_landsat.nir08 + imagen_landsat.red
)
ndvi_refl = (imagen_landsat_refl.nir08 - imagen_landsat_refl.red) / (
    imagen_landsat_refl.nir08 + imagen_landsat_refl.red
)

# Visualización comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# NDVI con números digitales
im1 = ax1.imshow(ndvi_dn, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
ax1.set_title("NDVI - Números Digitales")
plt.colorbar(im1, ax=ax1, fraction=0.046, label="NDVI")
ax1.axis("off")

# NDVI con reflectancia
im2 = ax2.imshow(ndvi_refl, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
ax2.set_title("NDVI - Reflectancia")
plt.colorbar(im2, ax=ax2, fraction=0.046, label="NDVI")
ax2.axis("off")

plt.suptitle("Comparación NDVI: Números Digitales vs Reflectancia", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

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
# Ahora si, calculamos NDVI y EVI para todo el stack temporal de Landsat:
# %%
ds_landsat_refl = ds_landsat * scale + offset

if len(items_landsat) > 0:
    # Calcular NDVI para todo el stack temporal
    ndvi_stack = (ds_landsat_refl.nir08 - ds_landsat_refl.red) / (
        ds_landsat_refl.nir08 + ds_landsat_refl.red
    )

    # Calcular EVI (Enhanced Vegetation Index)
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi_stack = (
        2.5
        * (ds_landsat_refl.nir08 - ds_landsat_refl.red)
        / (
            ds_landsat_refl.nir08
            + 6 * ds_landsat_refl.red
            - 7.5 * ds_landsat_refl.blue
            + 1
        )
    )

    # Agregar al dataset
    ds_landsat_refl = ds_landsat_refl.assign(NDVI=ndvi_stack, EVI=evi_stack)

    print(f"Índices calculados para {len(ds_landsat_refl.time)} fechas")
    print(f"Variables en el dataset: {list(ds_landsat_refl.data_vars)}")

    # Visualizar EVI de la fecha seleccionada
    evi_fecha = ds_landsat_refl.EVI.sel(time=fecha_landsat)

    plt.figure(figsize=(15, 10))
    im = plt.imshow(evi_fecha, cmap="RdYlGn", vmin=0, vmax=0.8)
    plt.colorbar(im, fraction=0.046, label="EVI")
    plt.title(
        f"Enhanced Vegetation Index (EVI)\nFecha: {fecha_landsat.values}", fontsize=14
    )
    plt.axis("off")
    plt.show()

# %% [markdown]
# ¿Cuál es la diferencia entre ambos objetos?
# %%
print(ds_landsat)
print(ds_landsat_refl)

# %% [markdown]
# ## 11. 🗺️ Extracción de Series Temporales con Mapa Interactivo

# %%
# @title Funciones utilitarias

current_ds = None

# Global variables for dynamic selection
selected_time_idx = 0
selected_display_mode = "RGB"


def create_rgb_image(ds, time_idx=0, bands=["red", "green", "blue"], enhance=3.0):
    """Create RGB image from xarray dataset"""
    rgb_data = []
    for band in bands:
        rgb_data.append(ds[band].isel(time=time_idx).values)

    rgb = np.stack(rgb_data, axis=-1)
    rgb_enhanced = np.clip(rgb * enhance, 0, 1)
    return rgb_enhanced


def create_selection_interface(ds):
    """
    CELL 1: Create dynamic selection interface
    Stores selections in global variables
    """
    global selected_time_idx, selected_display_mode, current_ds
    current_ds = ds

    # Create datetime options
    datetime_options = []
    for i, time_val in enumerate(ds.time.values):
        if hasattr(time_val, "strftime"):
            date_str = time_val.strftime("%Y-%m-%d %H:%M:%S")
        else:
            import pandas as pd

            date_str = pd.to_datetime(str(time_val)).strftime("%Y-%m-%d %H:%M:%S")
        datetime_options.append((date_str, i))

    # Create widgets
    time_dropdown = widgets.Dropdown(
        options=datetime_options,
        value=0,
        description="DateTime:",
        style={"description_width": "initial"},
    )

    display_dropdown = widgets.Dropdown(
        options=["RGB", "False Color NIR", "NDVI", "EVI"],
        value="RGB",
        description="Display Mode:",
        style={"description_width": "initial"},
    )

    # Status display
    status_output = widgets.Output()

    def update_selections():
        with status_output:
            clear_output(wait=True)
            selected_date = datetime_options[selected_time_idx][0]
            print(f"📅 Selected Date: {selected_date}")
            print(f"🎨 Selected Display: {selected_display_mode}")
            print("✅ Ready for plotting in next cell!")

    def on_time_change(change):
        global selected_time_idx
        selected_time_idx = change["new"]
        update_selections()

    def on_display_change(change):
        global selected_display_mode
        selected_display_mode = change["new"]
        update_selections()

    # Connect events
    time_dropdown.observe(on_time_change, names="value")
    display_dropdown.observe(on_display_change, names="value")

    # Display interface
    controls = widgets.VBox(
        [
            widgets.HTML("<h3>🎛️ Image Selection Interface</h3>"),
            time_dropdown,
            display_dropdown,
            widgets.HTML("<hr>"),
            widgets.HTML("<h4>📋 Current Selection:</h4>"),
            status_output,
        ]
    )

    display(controls)

    # Initial status update
    update_selections()

    print("🚀 Selection interface ready!")
    print("📝 Make your selections above, then run the plotting cell")


def plot_raster():
    """
    CELL 2: Plot image using selected parameters
    Simple plotting without click capture
    """
    global current_ds, selected_time_idx, selected_display_mode

    if current_ds is None:
        print("❌ No dataset loaded! Run the selection interface first.")
        return

    print(f"🎨 Plotting: {selected_display_mode}")
    print(f"📅 Date: {current_ds.time.values[selected_time_idx]}")
    print("-" * 50)

    # Create the plot based on selection
    if selected_display_mode == "RGB":
        img = create_rgb_image(current_ds, selected_time_idx, ["red", "green", "blue"])
        fig = px.imshow(img)
        fig.update_layout(title="True Color RGB")

    elif selected_display_mode == "False Color NIR":
        img = create_rgb_image(
            current_ds, selected_time_idx, ["nir08", "red", "green"], enhance=4.0
        )
        fig = px.imshow(img)
        fig.update_layout(title="False Color NIR-R-G")

    elif selected_display_mode == "NDVI":
        img_data = current_ds["NDVI"].isel(time=selected_time_idx)

        # Use percentiles for better color scaling
        p2, p98 = np.percentile(img_data.values[~np.isnan(img_data.values)], [2, 98])

        fig = px.imshow(
            img_data,
            color_continuous_scale="RdYlGn",
            zmin=max(p2, -0.2),
            zmax=min(p98, 1.0),
        )
        fig.update_layout(title="NDVI")

    else:  # EVI
        img_data = current_ds["EVI"].isel(time=selected_time_idx)

        # Use percentiles for better color scaling
        p2, p98 = np.percentile(img_data.values[~np.isnan(img_data.values)], [2, 98])

        fig = px.imshow(
            img_data,
            color_continuous_scale="RdYlGn",
            zmin=max(p2, -0.2),
            zmax=min(p98, 1.0),
        )
        fig.update_layout(title="EVI")

    fig.update_layout(width=800, height=600)
    fig.show()


def convert_coords_to_pixels(coord_list, ds):
    """
    Convert data coordinates to pixel indices
    """
    pixel_coords = []

    for i, (x_coord, y_coord) in enumerate(coord_list):
        # Find nearest pixel indices
        x_idx = np.argmin(np.abs(ds.x.values - x_coord))
        y_idx = np.argmin(np.abs(ds.y.values - y_coord))

        # Get actual coordinate values at those pixels
        actual_x = float(ds.x.isel(x=x_idx))
        actual_y = float(ds.y.isel(y=y_idx))

        pixel_coords.append(
            {
                "point_number": i + 1,
                "input_coords": (x_coord, y_coord),
                "actual_coords": (actual_x, actual_y),
                "pixel_indices": (x_idx, y_idx),
            }
        )

        print(
            f"Point {i + 1}: Input({x_coord}, {y_coord}) → Pixel({x_idx}, {y_idx}) → Actual({actual_x:.2f}, {actual_y:.2f})"
        )

    return pixel_coords


def plot_spectral_signatures(coord_list, band_name="NDVI", ds=None):
    """
    Plot spectral signatures (time series) for multiple coordinates using Plotly

    Parameters:
    - coord_list: List of [x, y] coordinate pairs
    - band_name: Band to plot ("NDVI", "EVI", "red", "green", "blue", etc.)
    - ds: Dataset (uses current_ds if None)
    """
    global current_ds

    if ds is None:
        ds = current_ds

    if ds is None:
        print("❌ No dataset available!")
        return

    if band_name not in ds.data_vars:
        print(f"❌ Band '{band_name}' not found in dataset!")
        print(f"Available bands: {list(ds.data_vars)}")
        return

    print(f"📈 Plotting spectral signatures for {band_name}")
    print(f"📍 Number of points: {len(coord_list)}")
    print("-" * 50)

    # Convert coordinates to pixels
    pixel_coords = convert_coords_to_pixels(coord_list, ds)

    # Create Plotly figure
    fig = go.Figure()

    # Color palette for different points
    colors = px.colors.qualitative.Set1[: len(coord_list)]
    if len(coord_list) > len(colors):
        colors = px.colors.qualitative.Plotly[: len(coord_list)]

    for i, coord_info in enumerate(pixel_coords):
        x_idx, y_idx = coord_info["pixel_indices"]
        input_x, input_y = coord_info["input_coords"]

        # Extract time series for this point
        point_ts = ds[band_name].isel(x=x_idx, y=y_idx)

        fig.add_trace(
            go.Scatter(
                x=ds.time.values,
                y=point_ts.values,
                mode="lines+markers",
                name=f"Point {i + 1}: ({input_x}, {input_y})",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"Spectral Signatures - {band_name}",
        xaxis_title="Date",
        yaxis_title=band_name,
        width=900,
        height=500,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    fig.show()
    print(f"✅ Spectral signatures plotted for {len(coord_list)} points")


def plot_multiple_bands_signatures(coord_list, bands=["NDVI", "EVI"], ds=None):
    """
    Plot spectral signatures for multiple bands in subplots using Plotly

    Parameters:
    - coord_list: List of [x, y] coordinate pairs
    - bands: List of band names to plot
    - ds: Dataset (uses current_ds if None)
    """
    global current_ds

    if ds is None:
        ds = current_ds

    if ds is None:
        print("❌ No dataset available!")
        return

    # Check if all bands exist
    missing_bands = [band for band in bands if band not in ds.data_vars]
    if missing_bands:
        print(f"❌ Bands not found: {missing_bands}")
        print(f"Available bands: {list(ds.data_vars)}")
        return

    print("📈 Plotting multi-band spectral signatures")
    print(f"📊 Bands: {bands}")
    print(f"📍 Number of points: {len(coord_list)}")
    print("-" * 50)

    # Convert coordinates to pixels
    pixel_coords = convert_coords_to_pixels(coord_list, ds)

    # Create subplots
    from plotly.subplots import make_subplots

    n_bands = len(bands)
    fig = make_subplots(
        rows=n_bands,
        cols=1,
        subplot_titles=[f"{band} Spectral Signatures" for band in bands],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Color palette for different points
    colors = px.colors.qualitative.Set1[: len(coord_list)]
    if len(coord_list) > len(colors):
        colors = px.colors.qualitative.Plotly[: len(coord_list)]

    for band_idx, band_name in enumerate(bands):
        for i, coord_info in enumerate(pixel_coords):
            x_idx, y_idx = coord_info["pixel_indices"]
            input_x, input_y = coord_info["input_coords"]

            # Extract time series for this point
            point_ts = ds[band_name].isel(x=x_idx, y=y_idx)

            fig.add_trace(
                go.Scatter(
                    x=ds.time.values,
                    y=point_ts.values,
                    mode="lines+markers",
                    name=f"Point {i + 1}: ({input_x}, {input_y})"
                    if band_idx == 0
                    else "",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    showlegend=(band_idx == 0),  # Only show legend for first subplot
                    legendgroup=f"point_{i}",  # Group legend items
                ),
                row=band_idx + 1,
                col=1,
            )

        # Update y-axis title for each subplot
        fig.update_yaxes(title_text=band_name, row=band_idx + 1, col=1)

    # Update x-axis title for bottom subplot
    fig.update_xaxes(title_text="Date", row=n_bands, col=1)

    fig.update_layout(
        height=300 * n_bands,
        width=900,
        title_text="Multi-Band Spectral Signatures",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    fig.show()
    print(f"✅ Multi-band spectral signatures plotted for {len(coord_list)} points")


def show_coordinates():
    """Display coordinates from my_coordinates list"""
    print("📍 Manual Coordinates:")
    print("=" * 40)
    for i, (x, y) in enumerate(my_coordinates):
        print(f"Point {i + 1}: X={x}, Y={y}")


def plot_time_series(band_name="NDVI"):
    """Legacy function - redirects to spectral signatures"""
    print("ℹ️  Use plot_spectral_signatures() instead")
    plot_spectral_signatures(my_coordinates, band_name)


# %%
create_selection_interface(ds_landsat_refl)

# %%
plot_raster()

# %%
my_coordinates = [[307965, 6217395], [323505, 6201135], [297585, 6223635]]
plot_spectral_signatures(my_coordinates, "NDVI")

# %%
plot_multiple_bands_signatures(my_coordinates, ["NDVI", "EVI"])
