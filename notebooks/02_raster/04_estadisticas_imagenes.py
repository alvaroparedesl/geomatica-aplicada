# %% [markdown]
# # Características y Estadísticas de Imágenes Satelitales
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/04_estadisticas_imagenes.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# En este cuaderno aprenderemos a analizar las características estadísticas de imágenes satelitales
# y a aplicar operadores lógicos para crear máscaras. Este es un cuaderno **práctico** con múltiples
# ejercicios para consolidar el aprendizaje.
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este cuaderno serás capaz de:
# * Calcular y visualizar histogramas de imágenes satelitales
# * Obtener estadísticas descriptivas por banda y por escena
# * Aplicar operadores lógicos para crear máscaras (AND, OR, NOT, XOR)
# * Integrar datos de elevación (DEM) con imágenes ópticas
# * Combinar múltiples condiciones para análisis avanzados

# %% [markdown]
# ## Configuración del entorno
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (buscar y cargar imágenes, convertir a reflectancia, histogramas, estadísticas, etc.).

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
# %pip install -q rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac scipy seaborn

# %%
# @title Importación de bibliotecas
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.colors import ListedColormap

import odc.stac

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import calcular_estadisticas, convertir_a_reflectancia
from utils.stac import (
    agregar_offset,
    cargar_dem,
    conectar_catalogo,
    parametros_busqueda,
    parametros_carga,
    preparar_items,
    renombrar_bandas,
    resumir_imagenes,
)
from utils.viz import crear_imagen_rgb, graficar_histogramas, mostrar_mascaras

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")
sns.set_palette("husl")

print("✅ Bibliotecas importadas exitosamente")

# %% [markdown]
# ## 1. Definición del Área de Estudio
#
# Trabajaremos con un área de 20km x 20km al sur de Santiago, en la zona del valle del Maipo.
# Esta área es interesante porque incluye:
# * Zonas urbanas
# * Agricultura
# * Vegetación natural
# * Ríos y cuerpos de agua
# * Variación topográfica (desde valle hasta cordillera)

# %%
# Área de interés: Valle del Maipo, al sur de Santiago
# Coordenadas aproximadas: Pirque, Puente Alto, San José de Maipo
bbox = [-70.8, -33.8, -70.6, -33.6]  # [oeste, sur, este, norte]

# Calculamos las dimensiones aproximadas
lon_diff = abs(bbox[2] - bbox[0])
lat_diff = abs(bbox[3] - bbox[1])

print("📍 Área de estudio: Valle del Maipo")
print(f"   Coordenadas: {bbox}")
print("   Dimensiones aproximadas:")
print(f"   - Longitud: {lon_diff * 111:.1f} km")  # 1 grado ≈ 111 km
print(f"   - Latitud: {lat_diff * 111:.1f} km")
print(f"   - Área total: ~{(lon_diff * 111) * (lat_diff * 111):.0f} km²")

# %% [markdown]
# ## 2. Búsqueda y Carga de Imágenes Sentinel-2
#
# Buscaremos imágenes de verano (enero-febrero) para tener buenas condiciones de iluminación
# y menor cobertura de nubes.

# %%
# Conectamos al catálogo STAC (cliente nativo de pystac_client, con enlaces firmados)
catalog = conectar_catalogo("planetary")

# Período de tiempo: verano 2024
fecha_inicio, fecha_fin = "2024-01-01", "2024-02-29"

# Búsqueda nativa: `parametros_busqueda` arma los argumentos (menos de 10% de nubes en la escena completa)
parametros = parametros_busqueda("sentinel-2", bbox, (fecha_inicio, fecha_fin), nubes_max=10, catalogo=catalog)
busqueda = catalog.search(**parametros)
items = busqueda.item_collection()

# Quitamos versiones repetidas de una misma toma
items = preparar_items(items)

# Tabla con fecha, % de nubes e identificador de cada imagen
print("\n📅 Imágenes disponibles:")
resumir_imagenes(items).head(10)

# %%
# Cargamos múltiples bandas para análisis estadístico completo
bandas_seleccionadas = [
    "B02",  # Azul (490 nm)
    "B03",  # Verde (560 nm)
    "B04",  # Rojo (665 nm)
    "B05",  # Red Edge 1 (705 nm)
    "B08",  # NIR (842 nm)
    "B11",  # SWIR 1 (1610 nm)
    "B12",  # SWIR 2 (2190 nm)
]

print(f"🔍 Cargando bandas: {bandas_seleccionadas}")

# Cargamos los datos con odc-stac (de forma perezosa: se descargan solo al usarlos).
# `parametros_carga` arma los argumentos de odc.stac.load; la carga es la llamada nativa.
parametros_ds = parametros_carga(
    "sentinel-2",
    bandas_seleccionadas,
    resolucion=10,  # 10m resolución para todas las bandas
    crs="EPSG:32719",  # UTM Zone 19S para Chile central
    catalogo=catalog,
    bbox=bbox,
    chunks={"x": 2048, "y": 2048},
)
ds = odc.stac.load(items, **parametros_ds)

# Nombres de banda estándar y offset de reflectancia de cada fecha
ds = renombrar_bandas(ds, catalogo=catalog)
ds = agregar_offset(ds, items)

print("\n📊 Dataset cargado:")
print(f"   Dimensiones: {dict(ds.sizes)}")
print(f"   Fechas disponibles: {len(ds.time)}")

# Seleccionamos la primera fecha y la descargamos a memoria (.compute())
fecha = ds.time[0]
imagen = ds.sel(time=fecha).compute()
print(f"   Trabajando con fecha: {str(fecha.values)[:10]}")

# %% [markdown]
# ## 3. Conversión a Reflectancia
#
# **¿Por qué convertir a reflectancia?**
#
# Los sensores satelitales guardan los datos como números digitales (DN). Para análisis
# científico correcto, debemos convertirlos a reflectancia, que representa la proporción
# de energía solar reflejada por la superficie (valores entre 0 y 1).
#
# Para Sentinel-2 L2A la conversión es:
#
# `reflectancia = DN * 0.0001 + offset`
#
# El factor 0.0001 es el mismo para todas las imágenes, pero el offset no: es -0.1 en las imágenes con
# versión de procesamiento (baseline) 04.00 o posterior, introducida el 25 de enero de 2022, que tienen un
# desfase de +1000 en los DN, y 0 en las anteriores. Depende del catálogo y del baseline de cada imagen,
# no solo de la fecha. Si lo olvidamos, la reflectancia queda sobreestimada en cerca de 0.1. Los helpers
# lo manejan por imagen (`agregar_offset` lo guarda en la coordenada `offset_reflectancia`) y con la
# función `convertir_a_reflectancia` no tenemos que acordarnos de estos valores.

# %%
# Conversión a reflectancia: reflectancia = DN * 0.0001 + offset  (offset de cada imagen, Sentinel-2 L2A)
# convertir_a_reflectancia también descarta (NaN) los valores fuera del rango 0 a 1.
imagen_refl = convertir_a_reflectancia(imagen, "sentinel-2")

print("🔄 Conversión a reflectancia completada")
print("\n📈 Comparación de rangos de valores:")
print("=" * 50)

for banda in bandas_seleccionadas:
    dn_min = imagen[banda].min().values
    dn_max = imagen[banda].max().values
    refl_min = imagen_refl[banda].min().values
    refl_max = imagen_refl[banda].max().values

    print(f"{banda}:")
    print(f"  DN:          {dn_min:6.0f} - {dn_max:6.0f}")
    print(f"  Reflectancia: {refl_min:.4f} - {refl_max:.4f}")

# %% [markdown]
# ## 4. Análisis de Histogramas
#
# Los histogramas nos muestran la distribución de valores de reflectancia en cada banda.
# Esto es fundamental para:
# * Entender las características espectrales del área
# * Identificar diferentes tipos de cobertura terrestre
# * Detectar problemas en los datos (saturación, nubes, sombras)


# %%
# Un histograma por banda, con la media y la mediana marcadas
graficar_histogramas(
    imagen_refl, bandas_seleccionadas, "Histogramas de Reflectancia - Sentinel-2", bins=50, columnas=3
)

# %% [markdown]
# ### Interpretación de Histogramas
#
# **¿Qué nos dicen los histogramas?**
#
# * **Bandas visibles (B02, B03, B04)**: Valores bajos típicos de superficies terrestres
# * **Banda NIR (B08)**: Valores más altos indican vegetación saludable
# * **Bandas SWIR (B11, B12)**: Útiles para detectar humedad y minerales
# * **Distribuciones bimodales**: Pueden indicar dos tipos de cobertura dominantes
# * **Colas largas**: Pueden indicar píxeles anómalos (nubes, agua, sombras)

# %% [markdown]
# ## 5. Estadísticas Descriptivas Completas
#
# Calculemos estadísticas detalladas para cada banda. Esto nos ayuda a caracterizar
# cuantitativamente nuestras imágenes.


# %%
# Estadísticas descriptivas por banda (media, mediana, desviación, cuartiles, asimetría, curtosis...)
df_stats = calcular_estadisticas(imagen_refl, bandas_seleccionadas)

print("📊 ESTADÍSTICAS DESCRIPTIVAS POR BANDA")
print("=" * 80)
print(df_stats.round(4).to_string(index=False))

# %% [markdown]
# ### Visualización de Estadísticas Clave

# %%
# Gráfico de caja (boxplot) para comparar distribuciones entre bandas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Preparar datos para boxplot
datos_boxplot = []
etiquetas_bandas = []

for banda in bandas_seleccionadas:
    datos = imagen_refl[banda].values.flatten()
    datos_validos = datos[~np.isnan(datos)]
    datos_boxplot.append(datos_validos)
    etiquetas_bandas.append(banda)

# Boxplot
ax1.boxplot(datos_boxplot)
ax1.set_xticklabels(etiquetas_bandas)
ax1.set_title("Distribución de Reflectancia por Banda")
ax1.set_ylabel("Reflectancia")
ax1.set_xlabel("Banda Espectral")
ax1.grid(True, alpha=0.3)

# Gráfico de medias y desviaciones estándar
medias = df_stats["Media"]
std_devs = df_stats["Desv_std"]
bandas_x = range(len(df_stats))

ax2.errorbar(bandas_x, medias, yerr=std_devs, marker="o", capsize=5, capthick=2)
ax2.set_xticks(bandas_x)
ax2.set_xticklabels(df_stats["Banda"])
ax2.set_title("Media ± Desviación Estándar por Banda")
ax2.set_ylabel("Reflectancia")
ax2.set_xlabel("Banda Espectral")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Operadores Lógicos y Creación de Máscaras
#
# Las máscaras son fundamentales en el análisis de imágenes satelitales. Nos permiten:
# * Seleccionar píxeles que cumplen ciertas condiciones
# * Excluir datos problemáticos (nubes, sombras, agua)
# * Enfocarnos en tipos específicos de cobertura terrestre
#
# ### Operadores Lógicos Básicos:
# * **AND (&)**: Ambas condiciones deben ser verdaderas
# * **OR (|)**: Al menos una condición debe ser verdadera
# * **NOT (~)**: Invierte la condición (verdadero ↔ falso)
# * **XOR (^)**: Solo una condición debe ser verdadera (exclusivo)

# %%
# Definimos umbrales para diferentes tipos de cobertura
print("🎯 CREANDO MÁSCARAS TEMÁTICAS")
print("=" * 40)

# Máscara de vegetación (NDVI alto)
ndvi = (imagen_refl.B08 - imagen_refl.B04) / (imagen_refl.B08 + imagen_refl.B04)
mascara_vegetacion = ndvi > 0.3

# Máscara de agua (baja reflectancia en NIR, alta en azul)
mascara_agua = (imagen_refl.B08 < 0.05) & (imagen_refl.B02 > 0.03)

# Máscara de suelo desnudo/urbano (alta reflectancia en SWIR)
# Umbrales de ejemplo: los elegimos mirando los histogramas de B11 y B12 (percentil 75 aprox.)
mascara_urbano = (imagen_refl.B11 > 0.30) & (imagen_refl.B12 > 0.20)

# Máscara de nubes (alta reflectancia en todas las bandas visibles)
mascara_nubes = (
    (imagen_refl.B02 > 0.20) & (imagen_refl.B03 > 0.20) & (imagen_refl.B04 > 0.20)
)

print(
    f"🌱 Vegetación:     {mascara_vegetacion.sum().values:6,} píxeles ({(mascara_vegetacion.sum() / mascara_vegetacion.size * 100).values:.1f}%)"
)
print(
    f"💧 Agua:           {mascara_agua.sum().values:6,} píxeles ({(mascara_agua.sum() / mascara_agua.size * 100).values:.1f}%)"
)
print(
    f"🏘️  Urbano/Suelo:   {mascara_urbano.sum().values:6,} píxeles ({(mascara_urbano.sum() / mascara_urbano.size * 100).values:.1f}%)"
)
print(
    f"☁️  Nubes:          {mascara_nubes.sum().values:6,} píxeles ({(mascara_nubes.sum() / mascara_nubes.size * 100).values:.1f}%)"
)

# %% [markdown]
# ### Aplicación de Operadores Lógicos

# %%
# Ejemplos prácticos de operadores lógicos
print("🔧 APLICANDO OPERADORES LÓGICOS")
print("=" * 40)

# AND: Vegetación Y no urbana (vegetación natural)
vegetacion_natural = mascara_vegetacion & (~mascara_urbano)

# OR: Agua O nubes (píxeles problemáticos)
pixeles_problematicos = mascara_agua | mascara_nubes

# XOR: Vegetación XOR urbano (áreas mutuamente excluyentes)
veg_xor_urbano = mascara_vegetacion ^ mascara_urbano

# NOT: Todo excepto agua
sin_agua = ~mascara_agua

print(
    f"🌿 Vegetación natural (AND):        {vegetacion_natural.sum().values:6,} píxeles"
)
print(
    f"⚠️  Píxeles problemáticos (OR):      {pixeles_problematicos.sum().values:6,} píxeles"
)
print(f"⚡ Vegetación XOR urbano:           {veg_xor_urbano.sum().values:6,} píxeles")
print(f"🚫 Sin agua (NOT):                  {sin_agua.sum().values:6,} píxeles")

# %% [markdown]
# ### Visualización de Máscaras


# %%
# Visualizar las máscaras principales (cada panel indica cuántos píxeles son True)
mascaras_principales = {
    "Vegetación": mascara_vegetacion,
    "Agua": mascara_agua,
    "Urbano/Suelo": mascara_urbano,
    "Nubes": mascara_nubes,
}

mostrar_mascaras(mascaras_principales)

# Visualizar operadores lógicos
mascaras_logicas = {
    "Vegetación Natural (AND)": vegetacion_natural,
    "Píxeles Problemáticos (OR)": pixeles_problematicos,
    "Vegetación XOR Urbano": veg_xor_urbano,
    "Sin Agua (NOT)": sin_agua,
}

mostrar_mascaras(mascaras_logicas)

# %% [markdown]
# ## 7. Integración con Datos de Elevación (DEM)
#
# Ahora integraremos datos de elevación del Copernicus DEM para análisis topográfico.
# Esto nos permitirá:
# * Analizar la relación entre elevación y cobertura terrestre
# * Crear máscaras basadas en topografía
# * Combinar criterios espectrales y topográficos

# %%
print("🏔️ CARGANDO DATOS DE ELEVACIÓN")
print("=" * 40)

# Cargamos el DEM Copernicus (30 m) del área, en el mismo CRS que la imagen Sentinel-2.
# Devuelve None si no hay datos para el área.
elevacion = cargar_dem(bbox, epsg="EPSG:32719", resolucion=30, catalogo=catalog)

if elevacion is not None:
    print("✅ DEM cargado exitosamente")
    print(f"   Dimensiones: {elevacion.shape}")
    print(f"   Elevación mínima: {elevacion.min().values:.0f} m")
    print(f"   Elevación máxima: {elevacion.max().values:.0f} m")
    print(f"   Elevación media: {elevacion.mean().values:.0f} m")

    # Re-muestrear el DEM (30 m) a la misma grilla que Sentinel-2 (10 m)
    elevacion_resample = elevacion.interp(
        x=imagen_refl.x, y=imagen_refl.y, method="linear"
    )
    print(f"   DEM re-muestreado a 10m: {elevacion_resample.shape}")

else:
    print("❌ No se encontraron datos de DEM")
    # Crear DEM sintético para propósitos de demostración
    print("📝 Creando DEM sintético para demostración...")
    x_coords = imagen_refl.x
    y_coords = imagen_refl.y

    # Crear gradiente sintético (elevación aumenta hacia el este)
    X, Y = np.meshgrid(x_coords, y_coords)
    elevacion_resample = xr.DataArray(
        500 + (X - X.min()) * 0.01 + np.random.normal(0, 50, X.shape),
        coords={"y": y_coords, "x": x_coords},
        dims=["y", "x"],
    )
    print(f"   DEM sintético creado: {elevacion_resample.shape}")

# %% [markdown]
# ### Visualización del DEM

# %%
# Visualizar el DEM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# DEM con colores topográficos
im1 = ax1.imshow(elevacion_resample, cmap="terrain")
ax1.set_title("Modelo Digital de Elevación (DEM)")
ax1.axis("off")
plt.colorbar(im1, ax=ax1, fraction=0.046, label="Elevación (m)")

# Histograma de elevaciones
elevacion_flat = elevacion_resample.values.flatten()
elevacion_valida = elevacion_flat[~np.isnan(elevacion_flat)]

ax2.hist(elevacion_valida, bins=50, alpha=0.7, color="brown", edgecolor="black")
ax2.set_title("Distribución de Elevaciones")
ax2.set_xlabel("Elevación (m)")
ax2.set_ylabel("Frecuencia")
ax2.grid(True, alpha=0.3)

# Estadísticas del DEM
media_elev = np.mean(elevacion_valida)
mediana_elev = np.median(elevacion_valida)
ax2.axvline(media_elev, color="red", linestyle="--", label=f"Media: {media_elev:.0f}m")
ax2.axvline(
    mediana_elev, color="blue", linestyle="--", label=f"Mediana: {mediana_elev:.0f}m"
)
ax2.legend()

plt.tight_layout()
plt.show()

print("📈 Estadísticas de elevación:")
print(f"   Rango: {elevacion_valida.min():.0f} - {elevacion_valida.max():.0f} m")
print(f"   Media: {media_elev:.0f} m ± {np.std(elevacion_valida):.0f} m")

# %% [markdown]
# ## 8. Combinación de Máscaras Espectrales y Topográficas
#
# Ahora combinaremos criterios espectrales con topográficos para crear análisis más sofisticados.

# %%
print("🔀 COMBINANDO CRITERIOS ESPECTRALES Y TOPOGRÁFICOS")
print("=" * 50)

# Definir rangos de elevación
elevacion_baja = elevacion_resample < np.percentile(elevacion_valida, 33)
elevacion_media = (elevacion_resample >= np.percentile(elevacion_valida, 33)) & (
    elevacion_resample < np.percentile(elevacion_valida, 67)
)
elevacion_alta = elevacion_resample >= np.percentile(elevacion_valida, 67)

# Convert the 0-dimensional NumPy arrays to scalar using .item()
print(
    f"🏞️ Elevación baja:  {elevacion_baja.sum().values.item():6} píxeles ({(elevacion_baja.sum() / elevacion_baja.size * 100).values.item():.1f}%)"
)
print(
    f"⛰️  Elevación media: {elevacion_media.sum().values.item():6} píxeles ({(elevacion_media.sum() / elevacion_media.size * 100).values.item():.1f}%)"
)
print(
    f"🏔️ Elevación alta:  {elevacion_alta.sum().values.item():6} píxeles ({(elevacion_alta.sum() / elevacion_alta.size * 100).values.item():.1f}%)"
)

# Combinaciones avanzadas
vegetacion_montana = mascara_vegetacion & elevacion_alta
vegetacion_valle = mascara_vegetacion & elevacion_baja
urbano_valle = mascara_urbano & elevacion_baja
agua_valle = mascara_agua & elevacion_baja

# Convert the 0-dimensional NumPy arrays to scalar using .item()
print(
    f"\n🌲 Vegetación de montaña:  {vegetacion_montana.sum().values.item():6,} píxeles"
)
print(f"🌾 Vegetación de valle:    {vegetacion_valle.sum().values.item():6,} píxeles")
print(f"🏘️ Urbano en valle:        {urbano_valle.sum().values.item():6,} píxeles")
print(f"🌊 Agua en valle:          {agua_valle.sum().values.item():6,} píxeles")

# %% [markdown]
# ### Análisis por Zonas de Elevación


# %%
# Análisis estadístico por zonas de elevación
def analizar_por_elevacion(imagen, elevacion_mask, zona_nombre):
    """
    Calcula estadísticas espectrales para una zona de elevación específica
    """
    print(f"\n📊 ZONA: {zona_nombre}")
    print("-" * 30)

    for banda in ["B04", "B08", "B11"]:  # Rojo, NIR, SWIR1
        # Aplicar máscara
        datos_zona = imagen[banda].where(elevacion_mask)
        datos_validos = datos_zona.values.flatten()
        datos_validos = datos_validos[~np.isnan(datos_validos)]

        if len(datos_validos) > 0:
            media = np.mean(datos_validos)
            std = np.std(datos_validos)
            print(f"  {banda}: {media:.4f} ± {std:.4f}")


# Análisis por zona
analizar_por_elevacion(imagen_refl, elevacion_baja, "ELEVACIÓN BAJA")
analizar_por_elevacion(imagen_refl, elevacion_media, "ELEVACIÓN MEDIA")
analizar_por_elevacion(imagen_refl, elevacion_alta, "ELEVACIÓN ALTA")

# %% [markdown]
# ### Visualización de Máscaras Combinadas

# %%
# Crear mapa de clasificación combinada
clasificacion = np.zeros_like(elevacion_resample.values)

# Asignar códigos de clase
clasificacion[vegetacion_montana] = 1  # Verde oscuro
clasificacion[vegetacion_valle] = 2  # Verde claro
clasificacion[urbano_valle] = 3  # Rojo
clasificacion[agua_valle] = 4  # Azul
clasificacion[mascara_nubes] = 5  # Blanco

# Crear máscara para píxeles no clasificados
sin_clasificar = clasificacion == 0
clasificacion[sin_clasificar] = 6  # Gris

# Definir colores personalizados
colores = ["darkgreen", "lightgreen", "red", "blue", "white", "gray"]
cmap_custom = ListedColormap(colores)

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# RGB original (rojo, verde, azul con brillo x3.5)
rgb = crear_imagen_rgb(imagen_refl, ["B04", "B03", "B02"], factor_brillo=3.5)

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Imagen Original RGB")
axes[0, 0].axis("off")

# DEM
im1 = axes[0, 1].imshow(elevacion_resample, cmap="terrain")
axes[0, 1].set_title("Modelo Digital de Elevación")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# Clasificación combinada
im2 = axes[1, 0].imshow(clasificacion, cmap=cmap_custom, vmin=1, vmax=6)
axes[1, 0].set_title("Clasificación Espectral-Topográfica")
axes[1, 0].axis("off")

# Crear leyenda personalizada
leyenda = [
    mpatches.Patch(color="darkgreen", label="Vegetación Montaña"),
    mpatches.Patch(color="lightgreen", label="Vegetación Valle"),
    mpatches.Patch(color="red", label="Urbano Valle"),
    mpatches.Patch(color="blue", label="Agua Valle"),
    mpatches.Patch(color="white", label="Nubes"),
    mpatches.Patch(color="gray", label="Sin Clasificar"),
]
axes[1, 0].legend(handles=leyenda, loc="center left", bbox_to_anchor=(1, 0.5))

# NDVI vs Elevación (scatter plot)
# Muestrear píxeles para visualización eficiente
mask_sample = np.random.random(elevacion_resample.shape) < 0.01  # 1% de los píxeles
elev_sample = elevacion_resample.values[mask_sample]
ndvi_sample = ndvi.values[mask_sample]

# Remover NaN
mask_valid = ~(np.isnan(elev_sample) | np.isnan(ndvi_sample))
elev_clean = elev_sample[mask_valid]
ndvi_clean = ndvi_sample[mask_valid]

axes[1, 1].scatter(elev_clean, ndvi_clean, alpha=0.6, s=1)
axes[1, 1].set_xlabel("Elevación (m)")
axes[1, 1].set_ylabel("NDVI")
axes[1, 1].set_title("Relación NDVI vs Elevación")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calcular correlación
correlacion = np.corrcoef(elev_clean, ndvi_clean)[0, 1]
print(f"🔗 Correlación NDVI-Elevación: {correlacion:.3f}")

# %% [markdown]
# ## 9. Ejercicios Prácticos
#
# **¡Hora de poner en práctica lo aprendido!** 💪
#
# Completa los siguientes ejercicios para consolidar tu comprensión:

# %% [markdown]
# ### Ejercicio 1: Análisis de Bandas Adicionales 📊
#
# **Objetivo**: Agregar las bandas B06 y B07 (Red Edge) al análisis
#
# **Tareas**:
# 1. Modifica la lista `bandas_seleccionadas` para incluir B06 y B07
# 2. Recarga el dataset con las nuevas bandas
# 3. Calcula estadísticas para las bandas Red Edge
# 4. Crea histogramas que incluyan estas bandas
# 5. Analiza qué información adicional proporcionan estas bandas

# %%
# ESPACIO PARA EJERCICIO 1
# Escribe tu código aquí:

# Tu código aquí...
print("🚧 Completa el Ejercicio 1")

# %% [markdown]
# ### Ejercicio 2: Máscaras Avanzadas 🎯
#
# **Objetivo**: Crear máscaras más específicas usando múltiples criterios
#
# **Tareas**:
# 1. Crea una máscara para "agricultura irrigada" combinando:
#    - NDVI moderado (0.2 - 0.6)
#    - Elevación baja
#    - Cerca de fuentes de agua
# 2. Crea una máscara para "bosque nativo" usando:
#    - NDVI alto (> 0.6)
#    - Elevación media-alta
#    - Textura (variabilidad local en B08)
# 3. Calcula el área (en hectáreas) de cada clase

# %%
# ESPACIO PARA EJERCICIO 2
# Pista: 1 píxel de 10m = 100 m² = 0.01 ha

# Tu código aquí...
print("🚧 Completa el Ejercicio 2")

# %% [markdown]
# ### Ejercicio 3: Análisis de Gradientes Topográficos 📈
#
# **Objetivo**: Analizar cómo cambian las propiedades espectrales con la elevación
#
# **Tareas**:
# 1. Divide el DEM en 5 zonas de elevación (usando quintiles)
# 2. Para cada zona, calcula la media de NDVI, B04 y B08
# 3. Crea un gráfico de líneas mostrando cómo cambian estos valores con la elevación
# 4. Interpreta los resultados: ¿qué patrones observas?

# %%
# ESPACIO PARA EJERCICIO 3
# Pista: usa np.percentile() para definir los quintiles

# Tu código aquí...
print("🚧 Completa el Ejercicio 3")

# %% [markdown]
# ### Ejercicio 4: Operadores Lógicos Complejos 🧠
#
# **Objetivo**: Practicar combinaciones complejas de operadores lógicos
#
# **Tareas**:
# 1. Crea una máscara para "zonas de riesgo de erosión":
#    - (Elevación alta AND vegetación baja) OR (pendiente alta AND suelo desnudo)
# 2. Crea una máscara para "áreas óptimas para desarrollo urbano":
#    - Elevación baja AND (NOT agua) AND (NOT vegetación densa) AND pendiente baja
# 3. Calcula el porcentaje del área total que cumple cada criterio

# %%
# ESPACIO PARA EJERCICIO 4
# Pista: Para calcular pendiente, usa np.gradient() sobre el DEM

# Tu código aquí...
print("🚧 Completa el Ejercicio 4")

# %% [markdown]
# ### Ejercicio 5: Análisis Temporal (Desafío) ⏰
#
# **Objetivo**: Si tienes múltiples fechas, analiza cambios temporales
#
# **Tareas**:
# 1. Si el dataset tiene múltiples fechas, calcula NDVI para cada una
# 2. Identifica píxeles donde el NDVI cambió significativamente
# 3. Crea una máscara de "cambio de vegetación"
# 4. Analiza qué factores topográficos se relacionan con estos cambios

# %%
# ESPACIO PARA EJERCICIO 5
# Pista: Usa ds.time para trabajar con múltiples fechas

if len(ds.time) > 1:
    # Tu código aquí para análisis temporal...
    print("🚧 Completa el Ejercicio 5 - Análisis Temporal")
else:
    print("ℹ️ Solo hay una fecha disponible. Este ejercicio requiere múltiples fechas.")

# %% [markdown]
# ## 10. Resumen y Conclusiones 📋
#
# En este cuaderno has aprendido a:
#
# ✅ **Análisis Estadístico**:
# - Calcular histogramas y estadísticas descriptivas
# - Interpretar distribuciones de reflectancia
# - Comparar características espectrales entre bandas
#
# ✅ **Operadores Lógicos**:
# - Aplicar AND, OR, NOT, XOR para crear máscaras
# - Combinar múltiples criterios espectrales
# - Visualizar y cuantificar resultados
#
# ✅ **Integración de Datos**:
# - Combinar imágenes ópticas con datos de elevación
# - Crear clasificaciones basadas en criterios múltiples
# - Analizar relaciones entre topografía y cobertura terrestre
#
# ### Próximos Pasos 🚀
#
# En el siguiente cuaderno (05) practicarás con un ejercicio interactivo de histogramas de bandas Landsat
# y selección de regiones, y luego seguiremos con los índices de vegetación (06) y el análisis multitemporal (07).
#
# ### Recursos Adicionales 📚
#
# - [Sentinel-2 User Handbook](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
# - [Remote Sensing Digital Image Analysis](https://link.springer.com/book/10.1007/978-3-662-05283-9)
# - [Earth Engine Apps](https://www.earthengine.app/) para comparar con GEE

# %%
print("🎉 ¡Cuaderno completado!")
print("📈 Has dominado el análisis estadístico de imágenes satelitales")
print("🔧 Has aprendido a usar operadores lógicos para crear máscaras")
print("🗺️ Has integrado datos ópticos y topográficos")
print("\n💡 ¡Ahora completa los ejercicios para consolidar tu aprendizaje!")
