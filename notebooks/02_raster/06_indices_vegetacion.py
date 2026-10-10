# %% [markdown]
# # Índices de Vegetación y Álgebra de Imágenes
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/06_indices_vegetacion.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
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
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (buscar y cargar imágenes, convertir a reflectancia, calcular índices, mostrar paneles, etc.).

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
# @title Instalación de paquetes necesarios
# %pip install -q rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

# %%
# @title Importación de bibliotecas
import numpy as np
import odc.stac
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import (
    calcular_ndvi,
    calcular_evi,
    calcular_ndwi,
    convertir_a_reflectancia,
    nombres_bandas,
    obtener_configuracion_sensor,
)
from utils.stac import (
    agregar_offset,
    conectar_catalogo,
    parametros_busqueda,
    parametros_carga,
    preparar_items,
    renombrar_bandas,
    resumir_imagenes,
)
from utils.viz import crear_imagen_rgb, mostrar_imagen, mostrar_paneles

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
# ## 2. Características de cada sensor
#
# Cada sensor tiene características diferentes (nombres de bandas, factores de conversión a reflectancia),
# y las funciones de `utils` las manejan automáticamente según el sensor que elijamos en `CONFIG`:
#
# | Aspecto | Sentinel-2 | Landsat 8/9 |
# |---------|------------|-------------|
# | **Bandas RGB** | B04, B03, B02 | red, green, blue |
# | **Banda NIR** | B08 | nir08 |
# | **Factor reflectancia** | 0.0001 | 0.0000275 |
# | **Offset** | -0.1 o 0, según el baseline de cada imagen (ver abajo) | -0.2 |
# | **Resolución nativa** | 10m | 30m |
#
# La conversión a reflectancia es `reflectancia = DN * factor + offset`. En Sentinel-2 el offset no es fijo:
# es -0.1 en las imágenes con versión de procesamiento (baseline) 04.00 o posterior, introducida el 25 de
# enero de 2022, y 0 en las anteriores. Depende del catálogo y del baseline de cada imagen, no solo de la
# fecha; sin el offset correcto, la reflectancia queda sobreestimada en cerca de 0.1 y los índices se
# distorsionan. Los helpers de `utils.stac` lo resuelven imagen por imagen (lo veremos al cargar los datos).

# %%
# Obtenemos la configuración del sensor seleccionado (bandas, factores, colección STAC...)
config_sensor = obtener_configuracion_sensor(CONFIG["sensor"])

print(f"🛰️ Trabajando con: {config_sensor['descripcion']}")
print(f"   Colección: {config_sensor['coleccion']}")
print(f"   Reflectancia = DN * {config_sensor['factor_reflectancia']} + offset (offset de referencia: {config_sensor['offset']})")
print(f"   Bandas: {config_sensor['bandas']}")

# %% [markdown]
# ## 3. Búsqueda y Carga de Datos
#
# Ahora cargaremos imágenes satelitales usando la configuración definida.

# %% [markdown]
# El flujo usa las llamadas nativas de `pystac_client` y `odc.stac`; los helpers de `utils.stac` solo arman
# sus argumentos (`parametros_busqueda`, `parametros_carga`) y ordenan los datos (`preparar_items`,
# `renombrar_bandas`, `agregar_offset`).

# %%
# Conectamos al catálogo STAC. Landsat solo existe en Planetary Computer; Sentinel-2 usa el catálogo
# por defecto del curso (`CATALOGO_SENTINEL2` en utils/stac.py).
catalog = conectar_catalogo("planetary" if CONFIG["sensor"] == "landsat" else None)

# Buscamos imágenes del sensor elegido (búsqueda nativa con los parámetros armados por el helper)
parametros = parametros_busqueda(
    CONFIG["sensor"],
    CONFIG["bbox"],
    (CONFIG["fecha_inicio"], CONFIG["fecha_fin"]),
    nubes_max=CONFIG["max_nubes"],
    catalogo=catalog,
)
busqueda = catalog.search(**parametros)
items = busqueda.item_collection()

# Quitamos versiones repetidas de una misma toma (Sentinel-2) y ordenamos por fecha
items = preparar_items(items)

# Tabla con fecha, % de nubes e identificador de cada imagen (en Sentinel-2, también baseline y offset)
resumir_imagenes(items)

# %%
# Definimos las bandas necesarias para el análisis (nombres genéricos -> nombres del sensor)
bandas_requeridas = nombres_bandas(CONFIG["sensor"], ["red", "green", "blue", "nir", "swir1"])

print(f"🎯 Bandas a cargar: {bandas_requeridas}")

# IMPORTANTE: Sin chunks se carga todo en memoria
if CONFIG["cargar_en_memoria"]:
    chunks = None
    print("🧠 Cargando completamente en memoria")
else:
    chunks = {"x": 1024, "y": 1024}
    print("📦 Cargando con chunks para optimizar memoria")

# Cargamos los datos con la llamada nativa de odc.stac; `parametros_carga` arma los argumentos
parametros_ds = parametros_carga(
    CONFIG["sensor"],
    bandas_requeridas,
    resolucion=CONFIG["resolucion"],
    crs=CONFIG["epsg"],
    catalogo=catalog,
    bbox=CONFIG["bbox"],
    chunks=chunks,
)
ds = odc.stac.load(items, **parametros_ds)

# Nombres de banda estándar y offset de reflectancia de cada fecha (Landsat queda igual)
ds = renombrar_bandas(ds, catalogo=catalog)
ds = agregar_offset(ds, items)

print("\n📊 Dataset cargado:")
print(f"   Dimensiones: {dict(ds.sizes)}")
print(f"   Variables: {list(ds.data_vars)}")
print(f"   Fechas disponibles: {len(ds.time)}")

# %%
# Seleccionamos la imagen con mejor calidad (menor cobertura de nubes)
fecha_seleccionada = ds.time[0]
imagen_original = ds.sel(time=fecha_seleccionada)

print("🖼️ Imagen seleccionada para análisis:")
print(f"   Fecha: {str(fecha_seleccionada.values)[:10]}")
print(f"   Dimensiones: {imagen_original.sizes['y']} x {imagen_original.sizes['x']}")

# %% [markdown]
# ## 4. Conversión a Reflectancia y Visualización RGB

# %%
# Convertimos a reflectancia (DN * factor + offset de cada imagen; descarta valores fuera de 0 a 1)
imagen_refl = convertir_a_reflectancia(imagen_original, CONFIG["sensor"])

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

bandas_info = " | ".join(
    [f"{b}: {config_sensor['bandas'][b]}" for b in ["red", "green", "blue"]]
)
mostrar_imagen(
    rgb_image,
    f"Imagen {config_sensor['descripcion']} - Color Verdadero (RGB)\n"
    f"Fecha: {str(fecha_seleccionada.values)[:10]} | Resolución: {CONFIG['resolucion']}m\n"
    f"Bandas utilizadas: {bandas_info}",
)

# %%
# Visualizamos imagen en falso color (NIR-R-G)
falso_color_image = crear_imagen_rgb(imagen_refl, config_sensor["falso_color"])

mostrar_imagen(
    falso_color_image,
    f"Imagen {config_sensor['descripcion']} - Falso Color (NIR-R-G)\n"
    "La vegetación aparece en tonos rojos",
)

print("💡 En falso color: Vegetación = Rojo | Agua = Negro/Azul | Suelo = Marrón")

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
mostrar_paneles(
    [
        ("Suma: NIR + RED\n(Energía total reflejada)", suma_nir_red, {"cmap": "viridis"}),
        ("Resta: NIR - RED\n(Base del NDVI)", diferencia_nir_red, {"cmap": "RdYlGn"}),
        ("Producto: NIR × RED\n(Interacción espectral)", producto_nir_red, {"cmap": "plasma"}),
        (
            "Ratio: NIR ÷ RED\n(Índice espectral simple)",
            ratio_nir_red,
            {"cmap": "coolwarm", "vmin": 0, "vmax": 5},
        ),
    ],
    titulo="Álgebra de Imágenes: Operaciones Aritméticas",
)

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
mascara = {"cmap": "RdYlBu_r", "barra": False}

mostrar_paneles(
    [
        (
            f"NIR > 0.3\n(Vegetación vigorosa: {nir_alto.mean().values * 100:.1f}%)",
            nir_alto,
            mascara,
        ),
        (
            f"RED < 0.1\n(Baja absorción: {red_bajo.mean().values * 100:.1f}%)",
            red_bajo,
            mascara,
        ),
        (
            f"0.2 ≤ NIR ≤ 0.4\n(Vegetación moderada: {nir_medio.mean().values * 100:.1f}%)",
            nir_medio,
            mascara,
        ),
        (
            f"(NIR - RED) > 0.1\n(Contraste alto: {diferencia_significativa.mean().values * 100:.1f}%)",
            diferencia_significativa,
            mascara,
        ),
    ],
    titulo="Álgebra de Imágenes: Operadores Lógicos (Máscaras Binarias)",
)

print("💡 Azul = False (0) | Rojo = True (1) | Las máscaras binarias filtran datos")

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
mostrar_paneles(
    [
        (
            f"AND: (NIR > 0.3) & (RED < 0.1)\nVegetación óptima: {vegetacion_optima.mean().values * 100:.1f}%",
            vegetacion_optima,
            {"cmap": "RdYlGn", "barra": False},
        ),
        (
            f"OR: (NIR > 0.4) | (RED > 0.2)\nReflectancia extrema: {reflectancia_extrema.mean().values * 100:.1f}%",
            reflectancia_extrema,
            {"cmap": "RdYlBu_r", "barra": False},
        ),
        (
            f"NOT: ~(NIR > RED)\nNo vegetación: {no_vegetacion.mean().values * 100:.1f}%",
            no_vegetacion,
            {"cmap": "RdYlBu", "barra": False},
        ),
        (
            f"Múltiples condiciones\nVegetación específica: {vegetacion_compleja.mean().values * 100:.1f}%",
            vegetacion_compleja,
            {"cmap": "Greens", "barra": False},
        ),
    ],
    titulo="Álgebra de Imágenes: Operadores Booleanos",
)

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
#
# **Fórmulas:**
#
# * **NDVI** = (NIR - ROJO) / (NIR + ROJO)
# * **EVI** = 2.5 * (NIR - ROJO) / (NIR + 6 * ROJO - 7.5 * AZUL + 1)
# * **NDWI** = (NIR - SWIR1) / (NIR + SWIR1)
#
# Todos son álgebra de imágenes: exactamente las operaciones de la sección anterior.

# %% [markdown]
# ### 6.1 NDVI "a mano"
#
# Con las bandas `nir` y `red` que ya extrajimos, el NDVI es una sola línea:

# %%
# NDVI manual: (NIR - ROJO) / (NIR + ROJO)
ndvi_manual = (nir - red) / (nir + red)

print(f"NDVI manual: rango {np.nanmin(ndvi_manual):.3f} - {np.nanmax(ndvi_manual):.3f}")

# %% [markdown]
# ### 6.2 Índices con funciones
#
# Las funciones `calcular_ndvi`, `calcular_evi` y `calcular_ndwi` aplican las fórmulas anteriores eligiendo
# por nosotros los nombres de banda de cada sensor. Además evitan dividir por cero y descartan (NaN) los
# valores fuera del rango -1 a 1.

# %%
# Calculamos los tres índices principales
ndvi = calcular_ndvi(imagen_refl, CONFIG["sensor"])
evi = calcular_evi(imagen_refl, CONFIG["sensor"])
ndwi = calcular_ndwi(imagen_refl, CONFIG["sensor"])

print("🌱 Índices de vegetación calculados:")
print(f"   NDVI: rango {np.nanmin(ndvi):.3f} - {np.nanmax(ndvi):.3f}")
print(f"   EVI:  rango {np.nanmin(evi):.3f} - {np.nanmax(evi):.3f}")
print(f"   NDWI: rango {np.nanmin(ndwi):.3f} - {np.nanmax(ndwi):.3f}")

# Comprobamos que la función da lo mismo que nuestro cálculo manual
diferencia = abs(ndvi - ndvi_manual).max().values
print(f"\n🔍 Diferencia máxima entre NDVI de la función y NDVI manual: {diferencia:.6f}")

# %%
# Visualizamos los índices de vegetación
mostrar_paneles(
    [
        ("NDVI\n(Normalized Difference Vegetation Index)", ndvi, {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 1}),
        ("EVI\n(Enhanced Vegetation Index)", evi, {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 1}),
        ("NDWI\n(Normalized Difference Water Index)", ndwi, {"cmap": "RdYlBu", "vmin": -0.5, "vmax": 0.5}),
        ("RGB Original\n(para comparación)", rgb_image),
    ],
    titulo="Índices de Vegetación Calculados",
)

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
