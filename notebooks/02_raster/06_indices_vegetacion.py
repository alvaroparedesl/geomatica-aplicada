# %% [markdown]
# # Índices de Vegetación y Álgebra de Imágenes
#
# En este cuaderno aprenderemos a calcular índices de vegetación utilizando álgebra de imágenes
# con datos de Landsat y Sentinel-2. Trabajaremos con **configuración modular** que permite
# cambiar fácilmente el sensor, área de estudio, resolución y período temporal.
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este cuaderno serás capaz de:
# * Configurar análisis para diferentes sensores (Landsat/Sentinel-2)
# * Aplicar álgebra de imágenes: operaciones aritméticas y lógicas
# * Crear y aplicar máscaras binarias
# * Calcular índices de vegetación (NDVI, EVI, NDWI)
# * Comparar índices usando visualizaciones interactivas
# * Interpretar resultados en contexto de recursos naturales

# %% [markdown]
# ## Configuración del entorno

# %%
# @title Instalación de paquetes necesarios
# %pip install rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

# %%
# @title Importación de bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from pystac_client import Client

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# Configuración para Colab

try:
    # Verificamos si estamos en Colab
    import google.colab

    IN_COLAB = True
    print("🔧 Configurado para Google Colab")
except ImportError:
    IN_COLAB = False
    print("🔧 Ejecutándose localmente")

print("✅ Bibliotecas importadas exitosamente")

# %% [markdown]
# ## 1. Configuración Modular del Análisis
#
# **¿Por qué usar configuración modular?**
#
# Permite cambiar fácilmente parámetros clave sin modificar el código principal:
# * **Sensor**: Landsat vs Sentinel-2 (diferentes bandas y factores de escala)
# * **Área**: Cualquier región de interés
# * **Resolución**: Desde 10m hasta 30m según necesidades
# * **Tiempo**: Cualquier período temporal

# %%
# 🎛️ CONFIGURACIÓN PRINCIPAL - Modifica estos valores según tus necesidades
CONFIG = {
    # === ÁREA DE ESTUDIO ===
    "bbox": [
        -70.8,
        -33.7,
        -70.6,
        -33.6,
    ],  # Valle del Maipo (formato: [oeste, sur, este, norte])
    "epsg": "EPSG:32719",  # UTM Zone 19S para Chile central
    # === PARÁMETROS TEMPORALES ===
    "fecha_inicio": "2024-01-01",
    "fecha_fin": "2024-02-28",  # Período de verano
    # === SENSOR (CONFIGURABLE) ===
    "sensor": "landsat",  # Opciones: 'sentinel-2' o 'landsat'
    # === PARÁMETROS DE CARGA ===
    "resolucion": 30,  # Resolución en metros
    "max_nubes": 15,  # Porcentaje máximo de nubes
    "cargar_en_memoria": True,  # Sin chunks - todo en memoria
}

# Mostramos la configuración actual
print("🎯 Configuración del análisis:")
print("=" * 40)
for clave, valor in CONFIG.items():
    print(f"  {clave}: {valor}")

# %% [markdown]
# ## 2. Funciones Especializadas por Sensor
#
# Cada sensor tiene características diferentes que debemos manejar automáticamente:
#
# | Aspecto | Sentinel-2 | Landsat 8/9 |
# |---------|------------|-------------|
# | **Bandas RGB** | B04, B03, B02 | red, green, blue |
# | **Banda NIR** | B08 | nir08 |
# | **Factor reflectancia** | 0.0001 | 0.0000275 |
# | **Offset** | 0 | -0.2 |
# | **Resolución nativa** | 10m | 30m |


# %%
# @title Funciones utilitarias
def obtener_configuracion_sensor(sensor):
    """
    Retorna configuración específica para cada sensor satelital.

    Parámetros:
    -----------
    sensor : str
        Nombre del sensor ('sentinel-2' o 'landsat')

    Retorna:
    --------
    dict
        Diccionario con configuración específica del sensor
    """

    if sensor == "sentinel-2":
        return {
            # Nombres de bandas Sentinel-2
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
            "falso_color": ["B08", "B04", "B03"],  # NIR, Red, Green
            # Parámetros de conversión a reflectancia
            "factor_reflectancia": 0.0001,
            "offset": 0.0,
            "rango_valido": (0, 1),
            # Configuración de búsqueda
            "coleccion": "sentinel-2-l2a",
            "plataforma_filtro": None,
            # Información para el estudiante
            "descripcion": "Sentinel-2 L2A (ESA) - Resolución 10m",
        }

    elif sensor == "landsat":
        return {
            # Nombres de bandas Landsat
            "bandas": {
                "red": "red",
                "green": "green",
                "blue": "blue",
                "nir": "nir08",
                "swir1": "swir16",
                "swir2": "swir22",
            },
            "rgb_bandas": ["red", "green", "blue"],
            "falso_color": ["nir08", "red", "green"],
            # Parámetros de conversión a reflectancia (Collection 2)
            "factor_reflectancia": 0.0000275,
            "offset": -0.2,
            "rango_valido": (0, 1),
            # Configuración de búsqueda
            "coleccion": "landsat-c2-l2",
            "plataforma_filtro": ["landsat-8", "landsat-9"],
            # Información para el estudiante
            "descripcion": "Landsat 8/9 Collection 2 L2 (USGS) - Resolución 30m",
        }
    else:
        raise ValueError(
            f"Sensor '{sensor}' no soportado. Use 'sentinel-2' o 'landsat'"
        )


def convertir_a_reflectancia(imagen, config_sensor):
    """
    Convierte números digitales a reflectancia según el sensor.

    Parámetros:
    -----------
    imagen : xarray.Dataset
        Dataset con las bandas espectrales
    config_sensor : dict
        Configuración específica del sensor

    Retorna:
    --------
    xarray.Dataset
        Dataset con valores de reflectancia validados
    """

    # Aplicamos la fórmula específica del sensor
    factor = config_sensor["factor_reflectancia"]
    offset = config_sensor["offset"]

    # Conversión: reflectancia = (DN * factor) + offset
    imagen_refl = imagen * factor + offset

    # Validamos rangos de reflectancia
    rango_min, rango_max = config_sensor["rango_valido"]
    imagen_refl = imagen_refl.where(
        (imagen_refl >= rango_min) & (imagen_refl <= rango_max)
    )

    print("🔄 Conversión a reflectancia completada:")
    print(f"   Factor: {factor}, Offset: {offset}")
    print(f"   Rango válido: {rango_min} - {rango_max}")

    return imagen_refl


def calcular_ndvi(imagen, config_sensor):
    """
    Calcula el NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - RED) / (NIR + RED)

    Parámetros:
    -----------
    imagen : xarray.Dataset
        Dataset con bandas espectrales
    config_sensor : dict
        Configuración del sensor

    Retorna:
    --------
    xarray.DataArray
        NDVI calculado (-1 a +1)
    """
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    red = imagen[bandas["red"]]

    # Evitamos división por cero
    denominador = nir + red
    ndvi = xr.where(denominador != 0, (nir - red) / denominador, np.nan)

    # Validamos rango: valores fuera de [-1, 1] se convierten a NaN
    ndvi = xr.where((ndvi >= -1) & (ndvi <= 1), ndvi, np.nan)

    # Añadimos metadatos
    ndvi.name = "NDVI"
    ndvi.attrs = {
        "long_name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "range": "-1 to +1",
    }

    return ndvi


def calcular_evi(imagen, config_sensor):
    """
    Calcula el EVI (Enhanced Vegetation Index).

    EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))

    Parámetros:
    -----------
    imagen : xarray.Dataset
        Dataset con bandas espectrales
    config_sensor : dict
        Configuración del sensor

    Retorna:
    --------
    xarray.DataArray
        EVI calculado
    """
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    red = imagen[bandas["red"]]
    blue = imagen[bandas["blue"]]

    # Fórmula EVI
    denominador = nir + 6 * red - 7.5 * blue + 1
    evi = xr.where(denominador != 0, 2.5 * (nir - red) / denominador, np.nan)

    # Validamos rango: valores fuera de [-1, 1] se convierten a NaN
    evi = xr.where((evi >= -1) & (evi <= 1), evi, np.nan)

    # Añadimos metadatos
    evi.name = "EVI"
    evi.attrs = {
        "long_name": "Enhanced Vegetation Index",
        "formula": "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        "range": "-1 to +1",
    }

    return evi


def calcular_ndwi(imagen, config_sensor):
    """
    Calcula el NDWI (Normalized Difference Water Index).

    NDWI = (NIR - SWIR1) / (NIR + SWIR1)

    Parámetros:
    -----------
    imagen : xarray.Dataset
        Dataset con bandas espectrales
    config_sensor : dict
        Configuración del sensor

    Retorna:
    --------
    xarray.DataArray
        NDWI calculado
    """
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    swir1 = imagen[bandas["swir1"]]

    # Fórmula NDWI
    denominador = nir + swir1
    ndwi = xr.where(denominador != 0, (nir - swir1) / denominador, np.nan)

    # Validamos rango: valores fuera de [-1, 1] se convierten a NaN
    ndwi = xr.where((ndwi >= -1) & (ndwi <= 1), ndwi, np.nan)

    # Añadimos metadatos
    ndwi.name = "NDWI"
    ndwi.attrs = {
        "long_name": "Normalized Difference Water Index",
        "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "range": "-1 to +1",
    }

    return ndwi


def crear_imagen_rgb(dataset, bandas_rgb, factor_brillo=3.0):
    """
    Crea una imagen RGB lista para visualización.

    Parámetros:
    -----------
    dataset : xarray.Dataset
        Dataset con las bandas espectrales
    bandas_rgb : list
        Lista con nombres de bandas [R, G, B]
    factor_brillo : float
        Factor para mejorar el brillo de la imagen

    Retorna:
    --------
    numpy.ndarray
        Array RGB normalizado para visualización
    """
    # Extraemos las bandas RGB
    rgb_data = []
    for banda in bandas_rgb:
        rgb_data.append(dataset[banda].values)

    # Apilamos las bandas (y, x, bandas)
    rgb = np.stack(rgb_data, axis=2)

    # Aplicamos factor de brillo y normalizamos
    rgb_enhanced = np.clip(rgb * factor_brillo, 0, 1)

    return rgb_enhanced


print("🔧 Funciones utilitarias creadas exitosamente")

# %% [markdown]
# ## 3. Búsqueda y Carga de Datos
#
# Ahora cargaremos imágenes satelitales usando la configuración definida.

# %%
# Obtenemos la configuración del sensor seleccionado
config_sensor = obtener_configuracion_sensor(CONFIG["sensor"])

print(f"🛰️ Trabajando con: {config_sensor['descripcion']}")
print(f"   Colección: {config_sensor['coleccion']}")

# %%
# Conectamos al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Configuramos la búsqueda según el sensor
query_params = {
    "collections": [config_sensor["coleccion"]],
    "bbox": CONFIG["bbox"],
    "datetime": f"{CONFIG['fecha_inicio']}/{CONFIG['fecha_fin']}",
    "query": {"eo:cloud_cover": {"lt": CONFIG["max_nubes"]}},
}

# Filtro adicional para Landsat (plataforma específica)
if config_sensor["plataforma_filtro"]:
    query_params["query"]["platform"] = {"in": config_sensor["plataforma_filtro"]}

# Realizamos la búsqueda
search = catalog.search(**query_params)
items = list(search.items())

print("🔍 Búsqueda completada:")
print(f"   Imágenes encontradas: {len(items)}")

if len(items) == 0:
    print("❌ No se encontraron imágenes. Intenta:")
    print("   - Ampliar el rango de fechas")
    print("   - Aumentar el porcentaje máximo de nubes")
    print("   - Cambiar el área de estudio")
else:
    # Información de la primera imagen
    item = items[0]
    print("\n📅 Primera imagen disponible:")
    print(f"   Fecha: {item.datetime.strftime('%Y-%m-%d')}")
    print(f"   Cobertura de nubes: {item.properties['eo:cloud_cover']:.1f}%")
    print(f"   ID: {item.id}")

# %%
# Definimos las bandas necesarias para el análisis
bandas_requeridas = [
    config_sensor["bandas"]["red"],
    config_sensor["bandas"]["green"],
    config_sensor["bandas"]["blue"],
    config_sensor["bandas"]["nir"],
    config_sensor["bandas"]["swir1"],
]

print(f"🎯 Bandas a cargar: {bandas_requeridas}")

# Configuramos parámetros de carga
load_params = {
    "items": items,
    "bands": bandas_requeridas,
    "bbox": CONFIG["bbox"],
    "crs": CONFIG["epsg"],
    "resolution": CONFIG["resolucion"],
    "groupby": "solar_day",
}

# IMPORTANTE: Sin chunks para cargar todo en memoria
if not CONFIG["cargar_en_memoria"]:
    load_params["chunks"] = {"x": 1024, "y": 1024}
    print("📦 Cargando con chunks para optimizar memoria")
else:
    print("🧠 Cargando completamente en memoria")

# Cargamos los datos
ds = odc.stac.load(**load_params)

print("\n📊 Dataset cargado:")
print(f"   Dimensiones: {dict(ds.sizes)}")
print(f"   Variables: {list(ds.data_vars)}")
print(f"   Fechas disponibles: {len(ds.time)}")

# %%
# Seleccionamos la imagen con mejor calidad (menor cobertura de nubes)
fecha_seleccionada = ds.time[0]
imagen_original = ds.sel(time=fecha_seleccionada)

print("🖼️ Imagen seleccionada para análisis:")
print(f"   Fecha: {fecha_seleccionada.values}")
print(f"   Dimensiones: {imagen_original.sizes['y']} x {imagen_original.sizes['x']}")

# %% [markdown]
# ## 4. Conversión a Reflectancia y Visualización RGB

# %%
# Convertimos a reflectancia usando la función específica del sensor
imagen_refl = convertir_a_reflectancia(imagen_original, config_sensor)

# Verificamos rangos de reflectancia por banda
print("\n📈 Rangos de reflectancia por banda:")
print("=" * 50)
for banda in bandas_requeridas:
    datos = imagen_refl[banda]
    datos_validos = datos.where(~np.isnan(datos))

    if datos_validos.count() > 0:
        min_val = float(datos_validos.min())
        max_val = float(datos_validos.max())
        mean_val = float(datos_validos.mean())
        print(f"{banda:>8}: {min_val:.4f} - {max_val:.4f} (promedio: {mean_val:.4f})")
    else:
        print(f"{banda:>8}: Sin datos válidos")

# %%
# Visualizamos la imagen RGB (color verdadero)
rgb_image = crear_imagen_rgb(imagen_refl, config_sensor["rgb_bandas"])

plt.figure(figsize=(15, 10))
plt.imshow(rgb_image)
plt.title(
    f"Imagen {config_sensor['descripcion']} - Color Verdadero (RGB)\n"
    f"Fecha: {fecha_seleccionada.values} | Resolución: {CONFIG['resolucion']}m",
    fontsize=14,
)
plt.axis("off")

# Añadimos información de las bandas
bandas_info = " | ".join(
    [f"{b}: {config_sensor['bandas'][b]}" for b in ["red", "green", "blue"]]
)
plt.figtext(
    0.5,
    0.02,
    f"Bandas utilizadas: {bandas_info}",
    ha="center",
    fontsize=10,
    style="italic",
)
plt.tight_layout()
plt.show()

# %%
# Visualizamos imagen en falso color (NIR-R-G)
falso_color_image = crear_imagen_rgb(imagen_refl, config_sensor["falso_color"])

plt.figure(figsize=(15, 10))
plt.imshow(falso_color_image)
plt.title(
    f"Imagen {config_sensor['descripcion']} - Falso Color (NIR-R-G)\n"
    f"La vegetación aparece en tonos rojos",
    fontsize=14,
)
plt.axis("off")

# Información sobre falso color
plt.figtext(
    0.5,
    0.02,
    "💡 En falso color: Vegetación = Rojo | Agua = Negro/Azul | Suelo = Marrón",
    ha="center",
    fontsize=11,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Álgebra de Imágenes: Fundamentos Teóricos y Prácticos
#
# **¿Qué es el álgebra de imágenes?**
#
# Es el conjunto de operaciones matemáticas que podemos aplicar a las imágenes satelitales:
# * **Aritméticas**: +, -, *, /
# * **Lógicas**: >, <, ==, !=, >=, <=
# * **Booleanas**: and (&), or (|), not (~)
#
# **¿Por qué es importante?**
# * Base para calcular índices espectrales
# * Crear máscaras para filtrar datos
# * Combinar información de múltiples bandas
# * Detectar características específicas del paisaje

# %%
# Extraemos bandas individuales para los ejemplos
nir = imagen_refl[config_sensor["bandas"]["nir"]]
red = imagen_refl[config_sensor["bandas"]["red"]]
green = imagen_refl[config_sensor["bandas"]["green"]]

print("🎯 Bandas extraídas para álgebra de imágenes:")
print(f"   NIR:   {config_sensor['bandas']['nir']} (infrarrojo cercano)")
print(f"   RED:   {config_sensor['bandas']['red']} (rojo)")
print(f"   GREEN: {config_sensor['bandas']['green']} (verde)")

# %% [markdown]
# ### 5.1 Operaciones Aritméticas Básicas

# %%
# Ejemplo 1: SUMA - Combinar señales espectrales
suma_nir_red = nir + red

# Ejemplo 2: RESTA - Base para índices de vegetación
diferencia_nir_red = nir - red

# Ejemplo 3: MULTIPLICACIÓN - Interacción entre bandas
producto_nir_red = nir * red

# Ejemplo 4: DIVISIÓN - Ratios espectrales
ratio_nir_red = xr.where(red != 0, nir / red, np.nan)

print("➕ Operaciones aritméticas calculadas:")
print(
    f"   Suma (NIR + RED): rango {suma_nir_red.min().values:.3f} - {suma_nir_red.max().values:.3f}"
)
print(
    f"   Resta (NIR - RED): rango {diferencia_nir_red.min().values:.3f} - {diferencia_nir_red.max().values:.3f}"
)
print(
    f"   Producto (NIR × RED): rango {producto_nir_red.min().values:.3f} - {producto_nir_red.max().values:.3f}"
)
print(
    f"   Ratio (NIR ÷ RED): rango {np.nanmin(ratio_nir_red):.3f} - {np.nanmax(ratio_nir_red):.3f}"
)

# %%
# Visualizamos las operaciones aritméticas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Álgebra de Imágenes: Operaciones Aritméticas", fontsize=16, fontweight="bold"
)

# Suma
im1 = axes[0, 0].imshow(suma_nir_red, cmap="viridis")
axes[0, 0].set_title("Suma: NIR + RED\n(Energía total reflejada)")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

# Resta
im2 = axes[0, 1].imshow(diferencia_nir_red, cmap="RdYlGn")
axes[0, 1].set_title("Resta: NIR - RED\n(Base del NDVI)")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# Multiplicación
im3 = axes[1, 0].imshow(producto_nir_red, cmap="plasma")
axes[1, 0].set_title("Producto: NIR × RED\n(Interacción espectral)")
axes[1, 0].axis("off")
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

# División
im4 = axes[1, 1].imshow(ratio_nir_red, cmap="coolwarm", vmin=0, vmax=5)
axes[1, 1].set_title("Ratio: NIR ÷ RED\n(Índice espectral simple)")
axes[1, 1].axis("off")
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Operadores de Comparación (Lógicos)

# %%
# Operadores lógicos crean imágenes binarias (True/False = 1/0)

# Ejemplo 1: Píxeles con alta reflectancia NIR (vegetación vigorosa)
nir_alto = nir > 0.3

# Ejemplo 2: Píxeles con baja reflectancia roja (menor absorción)
red_bajo = red < 0.1

# Ejemplo 3: Rango específico de NIR (vegetación moderada)
nir_medio = (nir >= 0.2) & (nir <= 0.4)

# Ejemplo 4: Diferencia significativa NIR-RED
diferencia_significativa = (nir - red) > 0.1

print("🔍 Operadores lógicos aplicados:")
print(
    f"   NIR alto (>0.3): {nir_alto.sum().values} píxeles ({nir_alto.mean().values * 100:.1f}%)"
)
print(
    f"   RED bajo (<0.1): {red_bajo.sum().values} píxeles ({red_bajo.mean().values * 100:.1f}%)"
)
print(
    f"   NIR medio (0.2-0.4): {nir_medio.sum().values} píxeles ({nir_medio.mean().values * 100:.1f}%)"
)
print(
    f"   Diferencia alta: {diferencia_significativa.sum().values} píxeles ({diferencia_significativa.mean().values * 100:.1f}%)"
)

# %%
# Visualizamos los operadores lógicos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Álgebra de Imágenes: Operadores Lógicos (Máscaras Binarias)",
    fontsize=16,
    fontweight="bold",
)

# NIR alto
axes[0, 0].imshow(nir_alto, cmap="RdYlBu_r")
axes[0, 0].set_title(
    f"NIR > 0.3\n(Vegetación vigorosa: {nir_alto.mean().values * 100:.1f}%)"
)
axes[0, 0].axis("off")

# RED bajo
axes[0, 1].imshow(red_bajo, cmap="RdYlBu_r")
axes[0, 1].set_title(
    f"RED < 0.1\n(Baja absorción: {red_bajo.mean().values * 100:.1f}%)"
)
axes[0, 1].axis("off")

# NIR medio
axes[1, 0].imshow(nir_medio, cmap="RdYlBu_r")
axes[1, 0].set_title(
    f"0.2 ≤ NIR ≤ 0.4\n(Vegetación moderada: {nir_medio.mean().values * 100:.1f}%)"
)
axes[1, 0].axis("off")

# Diferencia significativa
axes[1, 1].imshow(diferencia_significativa, cmap="RdYlBu_r")
axes[1, 1].set_title(
    f"(NIR - RED) > 0.1\n(Contraste alto: {diferencia_significativa.mean().values * 100:.1f}%)"
)
axes[1, 1].axis("off")

plt.figtext(
    0.5,
    0.02,
    "💡 Azul = False (0) | Rojo = True (1) | Las máscaras binarias filtran datos",
    ha="center",
    fontsize=11,
    style="italic",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.3 Operadores Booleanos (AND, OR, NOT)

# %%
# Combinamos múltiples condiciones usando operadores booleanos

# AND (&): Ambas condiciones deben ser verdaderas
vegetacion_optima = (nir > 0.3) & (red < 0.1)

# OR (|): Al menos una condición debe ser verdadera
reflectancia_extrema = (nir > 0.4) | (red > 0.2)

# NOT (~): Invierte la condición
no_vegetacion = ~(nir > red)

# Combinación compleja: vegetación con condiciones específicas
vegetacion_compleja = (nir > 0.25) & (red < 0.15) & ((nir - red) > 0.1)

print("🔗 Operadores booleanos aplicados:")
print(f"   Vegetación óptima (AND): {vegetacion_optima.sum().values} píxeles")
print(f"   Reflectancia extrema (OR): {reflectancia_extrema.sum().values} píxeles")
print(f"   No vegetación (NOT): {no_vegetacion.sum().values} píxeles")
print(f"   Vegetación compleja: {vegetacion_compleja.sum().values} píxeles")

# %%
# Visualizamos operadores booleanos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Álgebra de Imágenes: Operadores Booleanos", fontsize=16, fontweight="bold"
)

# AND
axes[0, 0].imshow(vegetacion_optima, cmap="RdYlGn")
axes[0, 0].set_title(
    f"AND: (NIR > 0.3) & (RED < 0.1)\nVegetación óptima: {vegetacion_optima.mean().values * 100:.1f}%"
)
axes[0, 0].axis("off")

# OR
axes[0, 1].imshow(reflectancia_extrema, cmap="RdYlBu_r")
axes[0, 1].set_title(
    f"OR: (NIR > 0.4) | (RED > 0.2)\nReflectancia extrema: {reflectancia_extrema.mean().values * 100:.1f}%"
)
axes[0, 1].axis("off")

# NOT
axes[1, 0].imshow(no_vegetacion, cmap="RdYlBu")
axes[1, 0].set_title(
    f"NOT: ~(NIR > RED)\nNo vegetación: {no_vegetacion.mean().values * 100:.1f}%"
)
axes[1, 0].axis("off")

# Combinación compleja
axes[1, 1].imshow(vegetacion_compleja, cmap="Greens")
axes[1, 1].set_title(
    f"Múltiples condiciones\nVegetación específica: {vegetacion_compleja.mean().values * 100:.1f}%"
)
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Cálculo de Índices de Vegetación
#
# **¿Qué son los índices de vegetación?**
#
# Son fórmulas matemáticas que combinan bandas espectrales para resaltar características
# de la vegetación:
#
# * **NDVI**: Indica vigor y densidad de vegetación
# * **EVI**: Menos sensible a la saturación en áreas densas
# * **NDWI**: Detecta contenido de agua en plantas y superficies

# %%
# Calculamos los tres índices principales
ndvi = calcular_ndvi(imagen_refl, config_sensor)
evi = calcular_evi(imagen_refl, config_sensor)
ndwi = calcular_ndwi(imagen_refl, config_sensor)

print("🌱 Índices de vegetación calculados:")
print(f"   NDVI: rango {np.nanmin(ndvi):.3f} - {np.nanmax(ndvi):.3f}")
print(f"   EVI:  rango {np.nanmin(evi):.3f} - {np.nanmax(evi):.3f}")
print(f"   NDWI: rango {np.nanmin(ndwi):.3f} - {np.nanmax(ndwi):.3f}")

# %%
# Visualizamos los índices de vegetación
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Índices de Vegetación Calculados", fontsize=16, fontweight="bold")

# NDVI
im1 = axes[0, 0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=1)
axes[0, 0].set_title("NDVI\n(Normalized Difference Vegetation Index)")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

# EVI
im2 = axes[0, 1].imshow(evi, cmap="RdYlGn", vmin=-0.2, vmax=1)
axes[0, 1].set_title("EVI\n(Enhanced Vegetation Index)")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# NDWI
im3 = axes[1, 0].imshow(ndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
axes[1, 0].set_title("NDWI\n(Normalized Difference Water Index)")
axes[1, 0].axis("off")
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

# Comparación RGB
axes[1, 1].imshow(rgb_image)
axes[1, 1].set_title("RGB Original\n(para comparación)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Análisis Interactivo

# %%
# @title Histogramas interactivos de los índices
ndvi_f = ndvi.values.ravel()
evi_f = evi.values.ravel()
ndwi_f = ndwi.values.ravel()

fig = go.Figure()

# Añadimos los tres índices en el mismo gráfico
fig.add_trace(
    go.Histogram(
        x=ndvi_f,
        xbins=dict(start=-1, end=1, size=0.025),
        name="NDVI",
        marker_color="green",
        opacity=0.6,
        histnorm="probability",
    )
)

fig.add_trace(
    go.Histogram(
        x=evi_f,
        xbins=dict(start=-1, end=1, size=0.025),
        name="EVI",
        marker_color="red",
        opacity=0.6,
        histnorm="probability",
    )
)

fig.add_trace(
    go.Histogram(
        x=ndwi_f,
        xbins=dict(start=-1, end=1, size=0.025),
        name="NDWI",
        marker_color="blue",
        opacity=0.6,
        histnorm="probability",
    )
)

# Configuramos el layout
fig.update_layout(
    title="Distribución de Índices de Vegetación",
    xaxis_title="Valor del Índice",
    yaxis_title="Probabilidad",
    barmode="overlay",  # Superpone los histogramas
    height=600,
    legend=dict(x=0.7, y=0.9),
)

fig.show()

# %%
# Scatter plot interactivo: NDVI vs EVI
fig = px.scatter(
    x=ndvi_f[::100],  # Submuestreamos para mejor rendimiento
    y=evi_f[::100],
    color=ndwi_f[::100],
    title="Relación entre NDVI y EVI (coloreado por NDWI)",
    labels={"x": "NDVI", "y": "EVI", "color": "NDWI"},
    color_continuous_scale="RdYlBu",
)
fig.update_layout(width=800, height=600)
fig.show()

# %%
# Estadísticas descriptivas
estadisticas = pd.DataFrame(
    {
        "Índice": ["NDVI", "EVI", "NDWI"],
        "Media": [np.nanmean(ndvi_f), np.nanmean(evi_f), np.nanmean(ndwi_f)],
        "Mediana": [np.nanmedian(ndvi_f), np.nanmedian(evi_f), np.nanmedian(ndwi_f)],
        "Desv. Estándar": [np.nanstd(ndvi_f), np.nanstd(evi_f), np.nanstd(ndwi_f)],
        "Mínimo": [np.nanmin(ndvi_f), np.nanmin(evi_f), np.nanmin(ndwi_f)],
        "Máximo": [np.nanmax(ndvi_f), np.nanmax(evi_f), np.nanmax(ndwi_f)],
    }
)

print("📈 Estadísticas descriptivas de los índices:")
print("=" * 70)
print(estadisticas.round(4).to_string(index=False))

# %% [markdown]
# ## 8. Interpretación y Conclusiones
#
# **Interpretación de los índices:**
#
# * **NDVI alto (>0.6)**: Vegetación densa y vigorosa (bosques, cultivos sanos)
# * **NDVI medio (0.2-0.6)**: Vegetación moderada (pastizales, cultivos en crecimiento)
# * **NDVI bajo (<0.2)**: Suelo desnudo, agua, áreas urbanas
#
# * **EVI**: Similar al NDVI pero menos sensible a la saturación
# * **NDWI positivo**: Presencia de agua o vegetación con alto contenido hídrico
# * **NDWI negativo**: Suelo seco, vegetación estresada
