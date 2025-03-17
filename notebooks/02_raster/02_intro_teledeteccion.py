# %% [markdown]
# # Introducción a la Teledetección
#
# En este cuaderno aprenderemos los conceptos básicos de la teledetección y su importancia en el estudio de los recursos naturales.

# %% [markdown]
# ## Configuración del entorno
#
# Primero, instalemos las bibliotecas necesarias para trabajar con datos de teledetección.


# %%
# Instalación de paquetes necesarios
# %pip install matplotlib numpy rioxarray xarray rasterio
# %%
# Importación de bibliotecas

import matplotlib.pyplot as plt
import numpy as np

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# %% [markdown]
# ## 1. ¿Qué es la Teledetección?
#
# La teledetección es la ciencia de obtener información sobre objetos o áreas desde la distancia, típicamente desde satélites o aeronaves, sin entrar en contacto físico con ellos. Utiliza sensores para captar la radiación electromagnética reflejada o emitida por la superficie terrestre.

# %% [markdown]
# ![Proceso de teledetección](https://www.earthdata.nasa.gov/sites/default/files/imported/RemoteSensing.jpg)

# %% [markdown]
# ## 2. Componentes de un sistema de teledetección
#
# Un sistema de teledetección consta de varios componentes:
#
# 1. **Fuente de energía**: Proporciona energía electromagnética al área de interés (sol, radar, etc.)
# 2. **Atmósfera**: La energía interactúa con la atmósfera al viajar de la fuente al objetivo
# 3. **Objetivo**: Objetos o áreas que reflejan o emiten radiación electromagnética
# 4. **Sensor**: Dispositivo que capta la radiación electromagnética y la convierte en señal
# 5. **Sistema de recepción y procesamiento**: Recibe la señal del sensor y la procesa en imágenes
# 6. **Interpretación y análisis**: Extracción de información útil de las imágenes
# 7. **Aplicación**: Uso de la información extraída para la toma de decisiones

# %% [markdown]
# ## 3. El espectro electromagnético
#
# La teledetección utiliza diferentes regiones del espectro electromagnético para obtener información sobre la superficie terrestre.
# ![espectro electromagnetico](https://mynasadata.larc.nasa.gov/sites/default/files/inline-images/Electromagnetic_Spectrum_Diagram%20flipped_FINAL.png)

# %% [markdown]
# ### Regiones importantes para la teledetección ambiental:
#
# - **Visible (0.4-0.7 µm)**: Refleja el color que vemos con nuestros ojos
# - **Infrarrojo cercano (0.7-1.3 µm)**: Muy útil para estudiar la vegetación
# - **Infrarrojo medio (1.3-3.0 µm)**: Sensible al contenido de agua en plantas y suelos
# - **Infrarrojo térmico (3.0-14 µm)**: Detecta el calor emitido por la superficie
# - **Microondas (1 mm - 1 m)**: Puede penetrar nubes y, en algunos casos, la vegetación

# %% [markdown]
# ## 4. Tipos de sensores
#
# Existen dos tipos principales de sensores en teledetección:

# %% [markdown]
# ### 4.1 Sensores pasivos
#
# Captan la radiación natural reflejada o emitida por los objetos. Ejemplos:
# - Cámaras fotográficas
# - Radiómetros multiespectrales (Landsat, Sentinel-2)
# - Sensores hiperespectrales (AVIRIS, Hyperion)
# - Radiómetros térmicos (TIRS en Landsat)

# %% [markdown]
# ### 4.2 Sensores activos
#
# Emiten su propia energía y miden la radiación reflejada. Ejemplos:
# - Radar (Sentinel-1, RADARSAT)
# - LiDAR (Light Detection and Ranging)
# - Sonar (para aplicaciones submarinas)

# %% [markdown]
# ## 5. Plataformas de teledetección
#
# Los sensores pueden estar montados en diferentes plataformas:

# %%
# Creamos una visualización de las diferentes plataformas
fig, ax = plt.subplots(figsize=(12, 8))

# Definimos las alturas de las plataformas
plataformas = [
    (0, "Superficie terrestre"),
    (0.1, "Drones (10-120 m)"),
    (3, "Aviones (1-10 km)"),
    (20, "Aviones de gran altitud (15-20 km)"),
    (400, "Satélites de órbita baja (400-1000 km)"),
    (36000, "Satélites geoestacionarios (36,000 km)"),
]

# Colores para cada plataforma
colores = ["brown", "orange", "green", "blue", "purple", "red"]

# Creamos una escala logarítmica para la visualización
alturas_log = [np.log10(max(h + 0.01, 0.01)) for h, _ in plataformas]
max_altura_log = np.log10(40000)

# Dibujamos líneas para cada plataforma
for i, (altura, nombre) in enumerate(plataformas):
    altura_log = np.log10(max(altura + 0.01, 0.01))
    ax.plot([0, 10], [altura_log, altura_log], color=colores[i], linewidth=3)
    ax.text(10.5, altura_log, nombre, va="center", fontsize=12)

# Configuramos los ejes
ax.set_xlim(0, 20)
ax.set_ylim(0, np.log10(40000))
ax.set_yticks([np.log10(h + 0.01) for h, _ in plataformas])
ax.set_yticklabels([f"{int(h) if h >= 1 else h} km" for h, _ in plataformas])
ax.set_xticks([])
ax.set_title("Plataformas de Teledetección por Altitud")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Resoluciones en teledetección
#
# La calidad y utilidad de los datos de teledetección dependen de cuatro tipos de resolución:

# %% [markdown]
# ### 6.1 Resolución espacial
#
# Se refiere al tamaño del área más pequeña que puede ser distinguida en una imagen (tamaño del píxel).
#
# - **Alta resolución**: < 1 m (WorldView, GeoEye)
# - **Media resolución**: 10-30 m (Sentinel-2, Landsat)
# - **Baja resolución**: > 250 m (MODIS, AVHRR)

# %%
# Visualizamos diferentes resoluciones espaciales
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Creamos una imagen simple
img = np.zeros((100, 100, 3))
img[30:70, 30:70, 0] = 1  # Cuadrado rojo
img[40:60, 40:60, 1] = 1  # Cuadrado verde en el centro

# Diferentes resoluciones
resoluciones = [
    (100, 100, "Alta resolución (1 m)"),
    (20, 20, "Media resolución (5 m)"),
    (10, 10, "Baja resolución (10 m)"),
]

for i, (rows, cols, title) in enumerate(resoluciones):
    # Reducimos la resolución
    img_reducida = img.copy()
    if rows < 100:
        from skimage.transform import resize

        img_reducida = resize(img, (rows, cols, 3), anti_aliasing=True)
        img_reducida = resize(img_reducida, (100, 100, 3), anti_aliasing=False)

    axs[i].imshow(img_reducida)
    axs[i].set_title(title)
    axs[i].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Resolución espectral
#
# Se refiere al número y ancho de las bandas espectrales que el sensor puede captar.
#
# - **Pancromática**: Una sola banda ancha (blanco y negro)
# - **Multiespectral**: Varias bandas (típicamente 3-15)
# - **Hiperespectral**: Cientos de bandas estrechas y continuas

# %%
# Visualizamos diferentes resoluciones espectrales
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Definimos el rango de longitudes de onda
longitudes_onda = np.linspace(0.4, 2.5, 1000)  # 0.4-2.5 µm

# Simulamos una curva espectral para vegetación
reflectancia = np.zeros_like(longitudes_onda)
# Visible
reflectancia[(longitudes_onda >= 0.4) & (longitudes_onda < 0.5)] = 0.1  # Azul
reflectancia[(longitudes_onda >= 0.5) & (longitudes_onda < 0.6)] = 0.2  # Verde
reflectancia[(longitudes_onda >= 0.6) & (longitudes_onda < 0.7)] = 0.1  # Rojo
# NIR
reflectancia[(longitudes_onda >= 0.7) & (longitudes_onda < 1.3)] = 0.5
# SWIR
reflectancia[(longitudes_onda >= 1.3) & (longitudes_onda < 1.9)] = 0.3
reflectancia[(longitudes_onda >= 1.9) & (longitudes_onda < 2.5)] = 0.2

# Añadimos ruido para hacerlo más realista
np.random.seed(42)
reflectancia += np.random.normal(0, 0.02, size=reflectancia.shape)

# 1. Pancromático
axs[0].plot(longitudes_onda, reflectancia, "k-", alpha=0.3)
# Banda pancromática (0.5-0.9 µm)
pan_mask = (longitudes_onda >= 0.5) & (longitudes_onda <= 0.9)
axs[0].fill_between(
    longitudes_onda, 0, reflectancia, where=pan_mask, color="gray", alpha=0.7
)
axs[0].set_title("Resolución Pancromática (1 banda ancha)")
axs[0].set_ylabel("Reflectancia")

# 2. Multiespectral (Landsat-8 OLI)
axs[1].plot(longitudes_onda, reflectancia, "k-", alpha=0.3)
# Bandas de Landsat-8
bandas_landsat = [
    (0.43, 0.45, "Coastal/Aerosol", "purple"),
    (0.45, 0.51, "Azul", "blue"),
    (0.53, 0.59, "Verde", "green"),
    (0.64, 0.67, "Rojo", "red"),
    (0.85, 0.88, "NIR", "darkred"),
    (1.57, 1.65, "SWIR1", "brown"),
    (2.11, 2.29, "SWIR2", "orange"),
]
for inicio, fin, nombre, color in bandas_landsat:
    mask = (longitudes_onda >= inicio) & (longitudes_onda <= fin)
    axs[1].fill_between(
        longitudes_onda, 0, reflectancia, where=mask, color=color, alpha=0.7
    )
    axs[1].text((inicio + fin) / 2, 0.6, nombre, ha="center", fontsize=8)
axs[1].set_title("Resolución Multiespectral (7 bandas - Landsat-8 OLI)")
axs[1].set_ylabel("Reflectancia")

# 3. Hiperespectral
axs[2].plot(longitudes_onda, reflectancia, "k-", alpha=0.3)
# Simulamos 100 bandas estrechas
for i in range(100):
    inicio = 0.4 + i * 0.02
    fin = inicio + 0.015
    if fin > 2.5:
        break
    mask = (longitudes_onda >= inicio) & (longitudes_onda <= fin)
    color = plt.cm.jet(i / 100)
    axs[2].fill_between(
        longitudes_onda, 0, reflectancia, where=mask, color=color, alpha=0.7
    )
axs[2].set_title("Resolución Hiperespectral (100+ bandas estrechas)")
axs[2].set_xlabel("Longitud de onda (µm)")
axs[2].set_ylabel("Reflectancia")

# Configuración común
for ax in axs:
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.7, color="k", linestyle="--", alpha=0.3)  # Límite visible/NIR
    ax.text(0.55, 0.65, "Visible", ha="center")
    ax.text(1.6, 0.65, "Infrarrojo", ha="center")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.3 Resolución radiométrica
#
# Se refiere a la sensibilidad del sensor para detectar diferencias en la intensidad de la señal.
#
# - **Baja**: 8 bits (256 niveles) - Landsat 7 ETM+
# - **Media**: 12 bits (4,096 niveles) - Landsat 8 OLI
# - **Alta**: 16 bits (65,536 niveles) - Algunos sensores modernos

# %%
# Visualizamos diferentes resoluciones radiométricas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Creamos una imagen de gradiente
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
img = X.copy()

# Diferentes resoluciones radiométricas
resoluciones = [
    (2, "2 bits (4 niveles)"),
    (4, "4 bits (16 niveles)"),
    (8, "8 bits (256 niveles)"),
]

for i, (bits, title) in enumerate(resoluciones):
    # Reducimos la resolución radiométrica
    niveles = 2**bits
    img_reducida = np.floor(img * (niveles - 1)) / (niveles - 1)

    axs[i].imshow(img_reducida, cmap="gray")
    axs[i].set_title(title)
    axs[i].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 Resolución temporal
#
# Se refiere a la frecuencia con la que un sensor revisita la misma área.
#
# - **Alta**: Diaria o varias veces al día (MODIS, GOES)
# - **Media**: Semanal (Sentinel-2 combinado: 5 días)
# - **Baja**: Mensual o mayor (Algunos satélites comerciales)

# %%
# Visualizamos la resolución temporal de diferentes satélites
satelites = [
    ("GOES", 0.042, "Cada hora"),
    ("MODIS (Terra/Aqua)", 1, "Diaria"),
    ("Sentinel-2 (A+B)", 5, "Cada 5 días"),
    ("Landsat-8", 16, "Cada 16 días"),
    ("Landsat-7", 16, "Cada 16 días"),
    ("WorldView-4", 4.5, "Variable ~4.5 días"),
]

# Ordenamos por resolución temporal
satelites.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(10, 6))

# Creamos barras horizontales
y_pos = np.arange(len(satelites))
resoluciones = [sat[1] for sat in satelites]
nombres = [sat[0] for sat in satelites]
etiquetas = [sat[2] for sat in satelites]

# Colores según la frecuencia (más frecuente = verde, menos frecuente = rojo)
colores = plt.cm.RdYlGn_r(np.array(resoluciones) / max(resoluciones))

bars = ax.barh(y_pos, resoluciones, color=colores)
ax.set_yticks(y_pos)
ax.set_yticklabels(nombres)
ax.set_xlabel("Días entre revisitas")
ax.set_title("Resolución Temporal de Diferentes Satélites")

# Añadimos etiquetas
for i, bar in enumerate(bars):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        etiquetas[i],
        va="center",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Aplicaciones de la teledetección en recursos naturales
#
# La teledetección tiene numerosas aplicaciones en el estudio y gestión de recursos naturales:
#
# - **Monitoreo de bosques**: Deforestación, reforestación, incendios forestales
# - **Agricultura**: Estado de cultivos, estrés hídrico, estimación de rendimientos
# - **Recursos hídricos**: Calidad del agua, niveles de embalses, inundaciones
# - **Biodiversidad**: Mapeo de hábitats, seguimiento de cambios en ecosistemas
# - **Cambio climático**: Monitoreo de glaciares, nivel del mar, temperatura superficial
# - **Gestión de desastres**: Evaluación de daños por terremotos, inundaciones, incendios

# %% [markdown]
# ## 8. Principales misiones satelitales para recursos naturales
#
# Algunas de las misiones satelitales más relevantes para el estudio de recursos naturales:

# %%
# Creamos una tabla con información sobre misiones satelitales
misiones = [
    (
        "Landsat",
        "NASA/USGS",
        "1972-presente",
        "30m (15m pan)",
        "16 días",
        "Multiespectral",
        "Monitoreo terrestre de larga duración",
    ),
    (
        "Sentinel-2",
        "ESA",
        "2015-presente",
        "10m, 20m, 60m",
        "5 días (A+B)",
        "Multiespectral",
        "Vegetación, suelos y aguas costeras",
    ),
    (
        "MODIS",
        "NASA",
        "1999-presente",
        "250m, 500m, 1km",
        "Diaria",
        "Multiespectral",
        "Cobertura global diaria",
    ),
    (
        "Sentinel-1",
        "ESA",
        "2014-presente",
        "5-40m",
        "6 días (A+B)",
        "Radar (SAR)",
        "Monitoreo en todas condiciones climáticas",
    ),
    (
        "SPOT",
        "CNES",
        "1986-presente",
        "1.5-6m",
        "1-5 días",
        "Multiespectral",
        "Cartografía y monitoreo ambiental",
    ),
]

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("tight")
ax.axis("off")

# Creamos la tabla
tabla = ax.table(
    cellText=[[m[0], m[1], m[2], m[3], m[4], m[5], m[6]] for m in misiones],
    colLabels=[
        "Misión",
        "Agencia",
        "Periodo",
        "Res. Espacial",
        "Res. Temporal",
        "Tipo",
        "Aplicaciones",
    ],
    loc="center",
    cellLoc="center",
)

# Ajustamos el tamaño de la tabla
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.5)

# Ajustamos el ancho de las columnas
for (i, j), cell in tabla.get_celld().items():
    if j == 6:  # Columna de aplicaciones
        cell.set_width(0.3)
    else:
        cell.set_width(0.1)

plt.title("Principales Misiones Satelitales para Recursos Naturales", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Resumen y conceptos clave
#
# En este cuaderno hemos aprendido:
#
# * Qué es la teledetección y sus componentes principales
# * El espectro electromagnético y su importancia en teledetección
# * Tipos de sensores: pasivos y activos
# * Plataformas de teledetección: desde drones hasta satélites
# * Las cuatro resoluciones: espacial, espectral, radiométrica y temporal
# * Aplicaciones de la teledetección en recursos naturales
# * Principales misiones satelitales para el estudio de recursos naturales
#
# En el próximo cuaderno, profundizaremos en los principios físicos de la teledetección.

# %% [markdown]
# ## Ejercicios
#
# 1. Investiga y compara las características de Landsat 8 y Sentinel-2. ¿Cuál sería más adecuado para monitorear cambios en bosques nativos? Justifica tu respuesta.
#
# 2. Explica cómo las diferentes resoluciones (espacial, espectral, radiométrica y temporal) afectan el estudio de un fenómeno ambiental de tu interés.
#
# 3. Busca ejemplos de aplicaciones de teledetección en tu región o país. ¿Qué satélites y sensores se utilizan? ¿Qué problemas ambientales se están abordando?
#
# 4. Diseña un sistema de teledetección hipotético para monitorear la calidad del agua en lagos. ¿Qué resoluciones necesitarías? ¿Qué bandas espectrales serían más útiles?
