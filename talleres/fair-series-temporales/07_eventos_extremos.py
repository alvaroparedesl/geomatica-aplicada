# %% [markdown]
# # Detección de eventos extremos: umbrales clásicos y aprendizaje automático
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/07_eventos_extremos.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# **Bloque 7 del taller "Datos FAIR, series temporales satelitales y detección de anomalías con Python"** (unos 100 minutos).
#
# Hasta aquí calculaste anomalías: cuánto se aleja un valor de lo normal para esa época del año. Ahora damos el paso siguiente: **decidir cuándo una anomalía es un episodio extremo**. Lo haremos con la **temperatura de superficie de Landsat 8 y 9** en el agua de la bahía de Concepción y sus alrededores, con una serie de más de 13 años y una pregunta concreta: ¿hubo episodios anormalmente cálidos o fríos, cuáles fueron los más importantes y ocurre alguno ahora?
#
# Landsat no es un producto diseñado para la temperatura del mar: pasa **cada 8 días** (con los dos satélites), las **nubes** borran muchas fechas (unas 10 a 25 observaciones útiles por año) y la banda térmica es de **100 m**. Esas limitaciones son el corazón del bloque, porque son las de casi cualquier serie satelital real.
#
# En este cuaderno vas a:
#
# 1. **Calibrar** los números digitales de Landsat a temperatura (°C) y a reflectancia, y comprobar que coinciden con la tabla preparada.
# 2. Mirar una **serie irregular**: cuántas fechas útiles hay, qué huecos dejan las nubes y cómo **controlar la calidad** con la reflectancia.
# 3. Construir **paso a paso, en pandas legible, una climatología** que funciona con pocas observaciones (la idea de Hobday, adaptada).
# 4. Detectar **episodios** de observaciones anómalas seguidas, sin cruzar huecos largos.
# 5. Usar lo que Landsat tiene de especial: **temperatura y reflectancia en el mismo instante** (agua frente a tierra, turbidez).
# 6. Comparar los umbrales clásicos con **Isolation Forest** (aprendizaje automático), también con varias variables a la vez.
# 7. Mapear la **frecuencia de anomalías** en una malla de 500 m.
#
# | Parte | Minutos |
# |---|---|
# | Calibración y datos | 20 |
# | Serie irregular y control de calidad | 15 |
# | Climatología por ventana | 15 |
# | Episodios | 15 |
# | Agua, tierra y reflectancia | 10 |
# | Isolation Forest | 15 |
# | Mapa de frecuencia | 10 |

# %% [markdown]
# ## Configuración del entorno
#
# La celda `Preparar entorno` deja disponible la carpeta `utils/` del curso y la variable `DATOS`, la carpeta con los archivos de datos del taller. Esta vez usamos `utils/extremos.py` (control de calidad, climatología, episodios, Isolation Forest y la malla), `utils/series.py` (la tabla de Landsat y el guardado de resultados), `utils/stac.py` (los parámetros de carga) y `utils/raster.py` (la comprobación de la calibración). La lectura de los datos, las series de cada punto y la conexión al catálogo las escribes tú con pandas, xarray y `pystac_client`.

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
DATOS = RAIZ / "talleres" / "fair-series-temporales" / "datos"  # carpeta con los archivos de datos del taller

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
# @title Importación de bibliotecas
import warnings

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pyarrow.parquet as pq
import xarray as xr
from IPython.display import display
from pyproj import Transformer
from pystac_client import Client

# Funciones de apoyo del curso (carpeta utils/)
from utils.extremos import (aislamiento_forest_observaciones, anomalia_malla, anomalias_observaciones,
                            climatologia_malla, climatologia_ventana, controlar_calidad_agua, detectar_episodios,
                            episodios_por_punto, frecuencia_anomalias_malla, resumen_eventos, solapamiento)
from utils.series import (agregar_atributos_landsat, bbox_punto, cargar_ventanas_landsat, construir_series_landsat,
                          guardar_series)
from utils.stac import parametros_carga

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
ROJO, AZUL, GRIS = "#c0392b", "#2471a3", "#7f8c8d"

# %% [markdown]
# ## 1. Los datos: Landsat 8 y 9
#
# **Landsat 8** (2013) y **Landsat 9** (2021) son satélites del USGS y la NASA. Cada uno pasa por el mismo lugar cada **16 días**; juntos, cada **8**. Miden en bandas ópticas (**30 m**) y en el infrarrojo térmico (**100 m**, que el USGS entrega remuestreada a 30 m). El producto que usamos es **Collection 2 Level-2**: la reflectancia de superficie y la **temperatura de superficie** ya corregidas de la atmósfera, listas para leer desde Planetary Computer (colección `landsat-c2-l2`).
#
# Pero vienen como **números enteros** (DN, *digital numbers*), no como grados ni como reflectancia. Hay que **calibrarlos**:
#
# | Variable | Banda | Calibración |
# |---|---|---|
# | Temperatura de superficie | `lwir11` (ST_B10) | kelvin = DN x 0,00341802 + 149,0; °C = kelvin - 273,15 |
# | Reflectancia (verde, rojo, NIR, SWIR1) | `green`, `red`, `nir08`, `swir16` | reflectancia = DN x 0,0000275 - 0,2 |
#
# Un DN de 0 significa "sin dato". Y es **temperatura de la superficie** (suelo, copas, agua), no del aire; sobre el agua mide la temperatura de la **piel** de la superficie (no la del agua en profundidad), con errores de alrededor de 1 K o más que crecen con la humedad atmosférica y las nubes delgadas; no es un producto oceanográfico.
#
# ### 1.1 Calibrar una escena con tus propias manos
#
# Leemos una ventana pequeña (unos 400 m de lado) de una escena **despejada**: el centro de la bahía de Concepción el **13 de febrero de 2024**. Es la llamada nativa de `pystac_client` (el parámetro `modifier=planetary_computer.sign_inplace` firma los enlaces para poder leer los archivos) y de `odc.stac`; `parametros_carga` solo arma sus argumentos.

# %%
LON, LAT = -73.0000, -36.6900                              # centro de la bahía de Concepción
bbox = bbox_punto(LON, LAT, 200)                           # ventana de unos 400 x 400 m

catalogo = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
items = list(catalogo.search(collections=["landsat-c2-l2"], bbox=bbox, datetime="2024-02-13/2024-02-13").items())
print([(i.id, i.properties["platform"]) for i in items])

carga = parametros_carga("landsat", ["lwir11", "red", "nir08", "qa_pixel"], 30, "EPSG:32718", catalogo, bbox)
crudo = odc.stac.load(items, **carga).isel(time=0).compute()
crudo

# %% [markdown]
# Los valores son enteros (`uint16`). Mira los DN de la banda térmica y calibra a mano: primero a kelvin, después a grados Celsius. Con la banda roja, a reflectancia.

# %%
dn_termico = crudo["lwir11"]
print("DN térmico (mediana de la ventana):", int(dn_termico.median()), "| tipo:", dn_termico.dtype)

kelvin = dn_termico * 0.00341802 + 149.0
celsius = kelvin - 273.15
print(f"Temperatura: {float(kelvin.median()):.2f} K = {float(celsius.median()):.2f} °C")

reflectancia_roja = crudo["red"] * 0.0000275 - 0.2
reflectancia_nir = crudo["nir08"] * 0.0000275 - 0.2
print(f"Reflectancia roja: {float(reflectancia_roja.median()):.4f} | NIR: {float(reflectancia_nir.median()):.4f}")

# %% [markdown]
# Una temperatura de unos 15 °C y una reflectancia roja de menos de 0,01 son lo esperable en agua abierta: el agua absorbe casi toda la luz roja e infrarroja. Veamos qué pasa si **nos saltamos un paso**:

# %%
print(f"Sin calibrar, el DN dice {float(dn_termico.median()):.0f} (¿°C? ¿K?)")
print(f"Sin el offset de -0,2, la reflectancia roja sería {float((crudo['red'] * 0.0000275).median()):.3f}: "
      f"{float((crudo['red'] * 0.0000275).median()) / float(reflectancia_roja.median()):.0f} veces mayor")
print(f"Con DN * 0,00341802 y sin sumar 149, la temperatura sería {float((dn_termico * 0.00341802).median() - 273.15):.0f} °C")

# %% [markdown]
# El offset de la reflectancia (-0,2) es **mayor que la reflectancia misma del agua**, así que olvidarlo da valores absurdos. Por eso la reflectancia se calibra siempre con la fórmula de arriba (`DN * 0,0000275 - 0,2`), y para la temperatura el curso tiene la función `convertir_a_temperatura` en `utils.raster`. Comprobemos que hace lo mismo que nuestra cuenta manual:

# %%
from utils.raster import convertir_a_temperatura

dif_temperatura = float(np.abs(convertir_a_temperatura(dn_termico) - celsius).max())
print(f"Diferencia con la calibración manual de la temperatura: {dif_temperatura:.1e} °C")

# %% [markdown]
# ### 1.2 La tabla preparada del taller
#
# Hacer esto para 6 puntos y unas 390 fechas desde 2013 toma unos 30 minutos, así que esa parte está **preparada antes del taller** (el script `datos/preparar_landsat.py` repite los pasos que acabamos de ver, con la máscara de nubes y el resumen por punto). Para cada punto y fecha, la tabla trae la **mediana de una ventana circular de 200 m de radio**, ya calibrada y sin los píxeles con nubes, sombras o saturación. La leemos con `pd.read_parquet`. Los metadatos viajan dentro del archivo, como bytes (la clave `pandas` es de pandas y no nos interesa), y se leen con `pyarrow`. Como solo importa el **día** de la toma (la hora es siempre casi la misma, unas 14:40 UTC), quitamos la hora de la fecha una sola vez, aquí con `dt.normalize()`.

# %%
ruta = DATOS / "series_landsat_concepcion.parquet"
tabla = pd.read_parquet(ruta)
tabla["fecha"] = tabla["fecha"].dt.normalize()      # solo importa el día: la hora de la toma se descarta
print(f"{len(tabla)} filas, {tabla['id'].nunique()} puntos, del {tabla['fecha'].min().date()} al {tabla['fecha'].max().date()}")
display(tabla.drop(columns=["nombre", "lat", "lon"]).head(4).round(3))
metadatos = {k.decode(): v.decode() for k, v in pq.read_schema(ruta).metadata.items() if k != b"pandas"}
for clave, valor in metadatos.items():
    if clave in ("titulo", "fuente", "licencia", "periodo"):
        print(f"{clave:9s}: {valor}")

# %% [markdown]
# Las columnas `st_dn` (DN antes de calibrar) y `st_c` (°C) permiten comprobar la calibración de **toda** la tabla, y las dos escenas de febrero de 2024 que acabamos de calibrar a mano deben dar lo mismo que la fila del 13 de febrero:

# %%
print("Máxima diferencia entre st_c y la fórmula aplicada a st_dn:",
      f"{(tabla['st_dn'] * 0.00341802 + 149.0 - 273.15 - tabla['st_c']).abs().max():.1e} °C")

# Reproducir la fila del 13 de febrero de 2024 con el mismo código que preparó la tabla
punto_bahia = tabla[tabla["id"] == "bahia_concepcion"].iloc[:1][["id", "nombre", "lat", "lon", "tipo"]]
punto_bahia = punto_bahia.assign(lat=LAT, lon=LON)
ventana = odc.stac.load(items, **parametros_carga("landsat", ["green", "red", "nir08", "swir16", "lwir11", "qa_pixel", "qa_radsat"],
                                                  30, "EPSG:32718", catalogo, bbox)).compute()
ventana = agregar_atributos_landsat(ventana, items)
fila = construir_series_landsat({"bahia_concepcion": ventana}, punto_bahia, radio_m=200).iloc[0]
guardada = tabla[(tabla["id"] == "bahia_concepcion") & (tabla["fecha"].dt.date == pd.Timestamp("2024-02-13").date())].iloc[0]
print(f"Temperatura al recalcular: {fila['st_c']:.3f} °C | en la tabla: {guardada['st_c']:.3f} °C")
print(f"Reflectancia roja al recalcular: {fila['rojo']:.4f} | en la tabla: {guardada['rojo']:.4f}")

# %% [markdown]
# ## 2. Una serie irregular: pasos del satélite, nubes y control de calidad
#
# Elegimos el **centro de la bahía** y miramos la serie sin ningún análisis todavía. La serie de un punto se saca de la tabla filtrando por `id`, usando la fecha como índice y tomando la columna de la variable (`st_c`, la temperatura en °C). Es una `Series` de pandas con **todos los pasos del satélite**; las fechas con nubes quedan como `NaN` (no se descartan, así vemos los huecos).

# %%
PUNTO = "bahia_concepcion"                     # prueba "san_vicente", "biobio_pluma", "golfo_arauco" o "laguna_grande"
st = tabla[tabla["id"] == PUNTO].set_index("fecha")["st_c"].astype(float)
print(f"{len(st)} pasos del satélite, {st.notna().sum()} con dato ({st.notna().mean():.0%}); {st.isna().sum()} perdidos por nubes")
st.dropna().tail()

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(st.dropna().index, st.dropna(), "o-", ms=3, lw=0.6, color=AZUL)
ax.set_ylabel("temperatura de superficie (°C)")
ax.set_title("Bahía de Concepción: temperatura de superficie Landsat 8/9 (cada punto es una pasada sin nubes)")
plt.show()

# %% [markdown]
# Hay algo raro: algunos puntos caen **muy por debajo** del resto (4 o 5 °C más fríos que cualquier dato cercano). Antes de analizar, necesitamos saber **cuántos datos hay y cómo se distribuyen**.
#
# ### 2.1 ¿Cuántas observaciones y con qué huecos?

# %%
por_anio = st.groupby(st.index.year).agg(pasos="size", con_dato=lambda s: int(s.notna().sum()))
por_mes = st.groupby(st.index.month).apply(lambda s: s.notna().mean())
intervalos = st.dropna().index.to_series().diff().dt.days.dropna()

fig, ejes = plt.subplots(1, 3, figsize=(13, 3.4))
ejes[0].bar(por_anio.index, por_anio["pasos"], color=GRIS, label="pasos del satélite")
ejes[0].bar(por_anio.index, por_anio["con_dato"], color=AZUL, label="con dato")
ejes[0].set_title("Observaciones por año")
ejes[0].legend(fontsize=8)
ejes[1].bar(por_mes.index, por_mes * 100, color=AZUL)
ejes[1].set_xticks(range(1, 13), labels=list("EFMAMJJASOND"))
ejes[1].set_title("% de pasos con dato por mes")
ejes[2].hist(intervalos, bins=range(0, int(intervalos.max()) + 8, 8), color=AZUL, rwidth=0.9)
ejes[2].set_xlabel("días entre observaciones válidas")
ejes[2].set_title("Intervalos entre observaciones")
plt.tight_layout()
plt.show()
print(f"Mediana de {intervalos.median():.0f} días entre observaciones válidas, máximo {intervalos.max():.0f}")

# %% [markdown]
# Qué nos dicen los gráficos:
#
# * **Entre 10 y 18 observaciones útiles por año hasta 2021 y entre 20 y 25 desde 2022**, no las 45 que da el satélite. Desde 2022 hay más porque se suma **Landsat 9**.
# * **En invierno falta más dato** (los meses de lluvia y nubosidad): lo que sabemos de junio y julio viene de menos fechas que lo que sabemos de enero. Esto es un **sesgo de cielo despejado**: solo medimos cuando no hay nubes, y los días sin nubes no son una muestra al azar del clima.
# * Lo normal son **16 días entre observaciones**, pero hay huecos de **más de 100 días**.
#
# Con estos datos **no se puede aplicar tal cual el método de Hobday** (ola de calor marina = 5 días seguidos sobre el percentil 90, con una serie diaria y 30 años de base): no hay serie diaria ni 30 años. Hay que **adaptar la idea**.
#
# ### 2.2 Control de calidad: la reflectancia delata las nubes que la máscara no vio
#
# La máscara de Landsat (`qa_pixel`) marca las nubes de confianza alta. Una nube delgada o la neblina costera pueden quedar sin marcar y medirse como "agua muy fría". Pero tenemos algo a favor: **medimos temperatura y reflectancia a la vez**. El agua abierta tiene una firma clara (reflectancia roja y SWIR1 casi 0); una nube es brillante. Veamos las observaciones más frías de los cinco puntos de agua junto a su reflectancia roja.

# %%
AGUA = ["bahia_concepcion", "san_vicente", "biobio_pluma", "golfo_arauco", "laguna_grande"]
filas = []
for punto in AGUA:
    de_punto = tabla[tabla["id"] == punto].set_index("fecha")
    s = de_punto["st_c"].astype(float)
    a = anomalias_observaciones(s, climatologia_ventana(s))["anomalia"].dropna()
    ext = de_punto.loc[a.index, ["rojo", "ndwi", "nubes_escena"]].assign(anomalia=a, punto=punto)
    filas.append(ext)
frias = pd.concat(filas).nsmallest(8, "anomalia")
frias.round(3)

# %% [markdown]
# Las **cuatro observaciones más frías** (anomalías de -9 a -16 °C) tienen una reflectancia roja de 0,05 hasta 0,95 y una reflectancia SWIR1 de 0,04 a 0,4. El agua abierta absorbe casi toda la luz roja e infrarroja: su reflectancia roja es menor que 0,01 y la SWIR1 menor que 0,02. Esos valores son de nube. La regla `controlar_calidad_agua` las descarta: en puntos de agua, **reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02** (demasiado brillo) significa observación sospechosa. No toca los puntos de tierra, cuya reflectancia es otra. Se usa el brillo y no el NDWI porque sobre agua oscura la reflectancia NIR es a menudo ligeramente negativa y se convierte en NaN: el NDWI no se puede calcular en 1 de cada 8 observaciones de agua.

# %%
tabla_qc = controlar_calidad_agua(tabla)
descartadas = (tabla["st_c"].notna() & tabla_qc["st_c"].isna()).groupby(tabla["id"]).sum()
print("Observaciones con dato descartadas por sospecha de nube:")
print(descartadas.to_string())

st_qc = tabla_qc[tabla_qc["id"] == PUNTO].set_index("fecha")["st_c"].astype(float)
print(f"\n{PUNTO}: {st.notna().sum()} observaciones antes del control, {st_qc.notna().sum()} después")

# %% [markdown]
# **Descartar sin mirar es peligroso**: una regla demasiado dura borraría eventos reales. Esta es deliberadamente simple (solo quita lo que el agua **no puede** reflejar, por brillo) y no es perfecta: puede dejar pasar neblina delgada y puede descartar alguna floración real de algas o agua muy turbia (que también brilla). Quedan además observaciones sospechosas que ninguna regla simple resuelve (las veremos en las secciones 4 y 5, cuando **varios puntos coinciden** en la misma fecha).
#
# ### Tu turno 1: ¿qué pasa en otro punto?
#
# Cambia `PUNTO` (arriba) por `"laguna_grande"` y repite los gráficos de 2.1. ¿Tiene más o menos observaciones útiles? ¿Cuántas observaciones descartó el control de calidad?

# %% [markdown]
# ## 3. Climatología con observaciones irregulares: la idea de Hobday, adaptada
#
# Para decir que una temperatura es **anormal** hay que saber qué es **normal para esa época del año**. Hobday construye una curva de "normal" para cada día del año, juntando los valores de los días vecinos de todos los años. Aquí hacemos lo mismo, con una diferencia: no tenemos un valor por día, sino unas **10 a 25 observaciones al año**.
#
# Los parámetros que usaremos:
#
# | Parámetro | Hobday (serie diaria) | Esta versión (Landsat) |
# |---|---|---|
# | Periodo base | 1991 a 2020 (30 años) | 2013 a 2024 (12 años completos) |
# | Ventana para juntar observaciones | 11 días | **31 días** (15 antes y 15 después) |
# | Suavizado de la curva | media móvil de 31 días | media móvil de 15 días |
# | Umbral | percentil 90 (calor) y 10 (frío) | igual |
#
# La ventana se ensancha porque con 11 días juntaríamos unas 6 observaciones; con 31, unas 16 (de 7 a 22).

# %% [markdown]
# ### 3.1 Un calendario de 366 días
#
# La climatología es **una curva por día del año**. Los años bisiestos tienen 366 días, así que el día 100 no cae en la misma fecha todos los años. La convención de Hobday: el 29 de febrero es siempre el día 60 y, en los años no bisiestos, desde el 1 de marzo se suma 1. Con pandas se escribe en una línea: `dayofyear` da el día del año y le sumamos 1 (un `True`) cuando el año **no** es bisiesto (`~is_leap_year`) y el mes es marzo o posterior.

# %%
datos = st_qc.dropna().to_frame("st")
indice = datos.index
datos["dia"] = indice.dayofyear.to_numpy() + (~indice.is_leap_year & (indice.month >= 3)).astype(int)
datos.loc["2022-03-01":"2022-03-20"].head(3)

# %% [markdown]
# ### 3.2 Juntar las observaciones cercanas y calcular media y percentiles
#
# Para un día del año, por ejemplo el **1 de febrero** (día 32), juntamos **todas las observaciones del periodo base cuyo día del año esté a 15 días o menos**, de todos los años. De ese montón sacamos la media y los percentiles 10 y 90.

# %%
base = datos.loc["2013":"2024"]
dia_objetivo = 32
distancia = (base["dia"] - dia_objetivo + 183) % 366 - 183           # distancia circular: el día 366 es vecino del 1
junto = base.loc[distancia.abs() <= 15, "st"]
print(f"Observaciones a 15 días o menos del 1 de febrero: {len(junto)}, de {junto.index.year.nunique()} años")
print(f"Media {junto.mean():.2f} °C | p10 {np.percentile(junto, 10):.2f} °C | p90 {np.percentile(junto, 90):.2f} °C")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.scatter(junto.index.year + junto.index.dayofyear / 366, junto, color=AZUL)
ax.axhline(junto.mean(), color="black", label="media")
ax.axhline(np.percentile(junto, 90), color=ROJO, ls="--", label="p90")
ax.axhline(np.percentile(junto, 10), color=AZUL, ls="--", label="p10")
ax.set_ylabel("°C")
ax.set_title("Observaciones que entran en la climatología del 1 de febrero (a +-15 días)")
ax.legend()
plt.show()

# %% [markdown]
# Repetimos eso para los 366 días con un ciclo. Es lo que hace `climatologia_ventana`; abajo lo escribimos a mano y comprobamos que coinciden.

# %%
valores, dias = base["st"].to_numpy(), base["dia"].to_numpy()
media, p90, p10, cuenta = (np.full(366, np.nan) for _ in range(4))
for i in range(366):
    cerca = np.abs((dias - (i + 1) + 183) % 366 - 183) <= 15
    cuenta[i] = cerca.sum()
    if cerca.sum() >= 8:                                                # con menos observaciones no hay climatología
        media[i], p90[i], p10[i] = valores[cerca].mean(), np.percentile(valores[cerca], 90), np.percentile(valores[cerca], 10)


def suavizar(curva, ancho=15):
    """Media móvil circular que ignora los NaN."""
    mitad = ancho // 2
    pegada = pd.Series(np.concatenate([curva[-mitad:], curva, curva[:mitad]]))
    return pegada.rolling(ancho, center=True, min_periods=1).mean().to_numpy()[mitad:-mitad]


clima_manual = pd.DataFrame({"media": suavizar(media), "p90": suavizar(p90), "p10": suavizar(p10), "n": cuenta.astype(int)},
                            index=pd.RangeIndex(1, 367, name="dia"))
clima = climatologia_ventana(st_qc)                                     # la versión de utils
print("Diferencia máxima con la versión de utils:",
      f"{np.nanmax(np.abs(clima[['media', 'p90', 'p10']].to_numpy() - clima_manual[['media', 'p90', 'p10']].to_numpy())):.2e} °C")
print(f"Observaciones por ventana: de {int(clima['n'].min())} a {int(clima['n'].max())} (mediana {int(clima['n'].median())})")

# %%
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.fill_between(clima.index, clima["p10"], clima["p90"], color=GRIS, alpha=0.25, label="entre p10 y p90")
ax.plot(clima["media"], color="black", label="media")
ax.plot(clima["p90"], color=ROJO, ls="--", label="p90 (umbral cálido)")
ax.plot(clima["p10"], color=AZUL, ls="--", label="p10 (umbral frío)")
ax.set_ylabel("temperatura de superficie (°C)")
ax.set_title("Climatología 2013-2024 de la bahía de Concepción (Landsat 8/9)")
ax.legend(ncol=2, fontsize=8)
ax2.fill_between(clima.index, clima["n"], color=AZUL, alpha=0.5)
ax2.axhline(8, color=ROJO, ls=":", label="mínimo para calcular (8)")
ax2.set_xlabel("día del año (calendario de 366 días)")
ax2.set_ylabel("observaciones")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# El agua de la bahía alcanza el máximo en verano (enero a marzo) y el mínimo en invierno (julio). Abajo se ve cuántas observaciones entran en cada día: de 7 a 22, bastante menos que las 330 de una climatología de Hobday con 30 años de datos diarios. Por eso las curvas son irregulares: el hundimiento brusco hacia el día 180 (fines de junio), donde hay muy pocas observaciones, es **ruido de muestreo** y no una caída real de la temperatura. Esa banda entre el p10 y el p90 es el "margen de lo normal" contra el que medimos cada observación.
#
# ### Tu turno 2: ¿qué tan sensible es la climatología a la ventana?
#
# Calcula la climatología con la ventana de Hobday (11 días) y con una más ancha (61 días). Compara cuántas observaciones junta cada una y qué tan quebrada queda la curva del p90.

# %%
for ventana in (11, 31, 61):
    c = climatologia_ventana(st_qc, ventana=ventana, minimo_obs=3)
    ruido = c["p90"].diff().abs().mean()
    print(f"Ventana de {ventana:2d} días: mediana de {int(c['n'].median()):2d} observaciones; "
          f"días con climatología: {int(c['media'].notna().sum())}; salto medio diario del p90: {ruido:.3f} °C")

# %% [markdown]
# ## 4. Anomalías y episodios
#
# ### 4.1 La anomalía de cada observación
#
# `anomalias_observaciones` pone al lado de cada observación su **media** y sus **umbrales** del día del año, y calcula la **anomalía** (valor menos media). Una observación **excede** si está sobre el p90 (cálida) o bajo el p10 (fría). Una fecha sin dato no excede nada.

# %%
anomalias = anomalias_observaciones(st_qc, clima)
print(f"Observaciones con dato: {anomalias['valor'].notna().sum()}")
print(f"Sobre el p90: {anomalias['excede_calor'].sum()} ({anomalias['excede_calor'].sum() / anomalias['valor'].notna().sum():.1%}) | "
      f"bajo el p10: {anomalias['excede_frio'].sum()} ({anomalias['excede_frio'].sum() / anomalias['valor'].notna().sum():.1%})")
anomalias.dropna(subset=["valor"]).tail(5).round(2)

# %% [markdown]
# Por construcción, cerca del 10 % de las observaciones del periodo base supera el p90. Una observación suelta no es un episodio: es una medida, que puede tener ruido o ser una nube que se coló. Hace falta que **persista**.
#
# ### 4.2 Qué es un episodio aquí
#
# En Hobday, una ola de calor son **5 días seguidos** sobre el p90. Con una pasada cada 8 a 16 días eso no existe. La versión por observaciones:
#
# * Un **episodio** son **al menos 2 observaciones válidas consecutivas** que superan el p90 (cálido) o quedan bajo el p10 (frío).
# * Las **fechas con nubes** (NaN) no cuentan como observación: ni cortan la racha ni la alargan.
# * Entre dos observaciones consecutivas del episodio pueden pasar **24 días como máximo** (tres pasadas): una racha **no cruza un hueco largo**, porque no sabemos qué pasó ahí.
# * Una observación válida que **no** excede (un dato normal en medio) **corta** la racha.
# * **No es un evento de Hobday**: la `duracion` (de la primera a la última observación) es solo una **cota mínima**, no sabemos qué pasó entre pasadas.
#
# **Una advertencia sobre el muestreo.** Antes de 2022 solo había Landsat 8: la mayoría de los intervalos entre observaciones válidas es de **16 días**. Desde 2022, con Landsat 9, es de **8 días**. La brecha máxima de 24 días permite entonces que **no haya ninguna** pasada nublada entre dos observaciones antes de 2022 y **hasta dos** después, y dos observaciones a 8 días están más correlacionadas que a 16. Además, la climatología 2013-2024 pesa más a 2022-2024, que aportan el doble de observaciones por año. Por todo esto **no se deben leer tendencias en el conteo de episodios**: con otro muestreo, el mismo clima da otros números.
#
# Probemos las reglas en **series de juguete**, donde sabemos la respuesta. Una climatología plana (media 0, umbrales +1 y -1) y fechas irregulares:

# %%
clima_plana = pd.DataFrame({"media": 0.0, "p90": 1.0, "p10": -1.0}, index=pd.RangeIndex(1, 367, name="dia"))


def juguete(*pares):
    """Serie irregular a partir de pares (fecha, valor); un valor None es una fecha con nubes (NaN)."""
    return pd.Series([np.nan if v is None else v for _, v in pares], index=pd.to_datetime([f for f, _ in pares]), dtype=float)


casos = {
    "2 sobre el umbral, 8 días entre ellas": juguete(("2021-03-01", 0), ("2021-03-09", 2), ("2021-03-17", 2), ("2021-03-25", 0)),
    "2 sobre el umbral, una fecha con nubes en medio": juguete(("2021-03-01", 2), ("2021-03-09", None), ("2021-03-17", 2)),
    "2 sobre el umbral, un dato normal en medio": juguete(("2021-03-01", 2), ("2021-03-09", 0), ("2021-03-17", 2)),
    "2 sobre el umbral, 24 días entre ellas": juguete(("2021-03-01", 2), ("2021-03-25", 2)),
    "2 sobre el umbral, 25 días entre ellas": juguete(("2021-03-01", 2), ("2021-03-26", 2)),
    "una sola observación sobre el umbral": juguete(("2021-03-01", 0), ("2021-03-09", 3), ("2021-03-17", 0)),
}
for nombre, serie in casos.items():
    ev = detectar_episodios(serie, clima_plana, "calor")
    print(f"{nombre:50s} -> {len(ev)} episodio(s)", [(int(n), int(d)) for n, d in zip(ev["n_obs"], ev["duracion"])])

# %% [markdown]
# Lee los resultados (entre paréntesis, observaciones y días):
#
# * **8 días entre dos observaciones**: 1 episodio de 2 observaciones y **9 días** (la cota mínima: de la primera a la última, ambas incluidas).
# * **Una fecha con nubes en medio**: 1 episodio, porque la nube no es un dato normal, es ausencia de dato.
# * **Un dato normal en medio**: ninguno. Ese dato demuestra que la temperatura volvió a lo normal.
# * **24 días** se une; **25 días** no: el hueco ya es demasiado largo.
# * **Una sola observación**: ninguna. Hace falta que persista.

# %% [markdown]
# ### 4.3 Los episodios de la bahía
#
# Ahora con los datos reales de la bahía, ya con control de calidad. El resultado es una tabla con las mismas columnas que la de eventos de Hobday (`inicio`, `fin`, `duracion`, `dia_pico`, `intensidad_max`, `intensidad_media`, `en_curso`) más `n_obs`.

# %%
calor = detectar_episodios(st_qc, clima, "calor")
frio = detectar_episodios(st_qc, clima, "frio")
print(f"{PUNTO}: {len(calor)} episodios cálidos y {len(frio)} fríos en {st_qc.index[0].year}-{st_qc.index[-1].year}")
columnas = ["inicio", "fin", "duracion", "n_obs", "dia_pico", "intensidad_max", "intensidad_media", "en_curso"]
display(pd.concat([calor, frio])[["tipo", *columnas]].round({"intensidad_max": 2, "intensidad_media": 2}).reset_index(drop=True))

# %% [markdown]
# Pocos episodios (en la bahía, ninguno cálido): con 10 a 25 observaciones al año y la exigencia de **dos seguidas**, solo se detectan los más persistentes. Eso **subestima** la frecuencia real: un episodio que cayó entre dos pasadas, o bajo una nube, no se ve. La tabla es un **mínimo**, y así hay que reportarla.
#
# Veámoslos en el tiempo: la línea es la anomalía de cada observación válida; las franjas rojas y azules son los episodios.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
valida = anomalias["anomalia"].dropna()
ax.plot(valida.index, valida, "o", ms=3, color="black")
for ev in calor.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), color=ROJO, alpha=0.5, lw=0)
for ev in frio.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), color=AZUL, alpha=0.5, lw=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlim(st_qc.index[0], st_qc.index[-1])
ax.set_ylabel("anomalía (°C)")
ax.set_title("Anomalía de cada observación y episodios (rojo: cálido, azul: frío)")
plt.show()

# %% [markdown]
# ### 4.4 Los cinco puntos de agua
#
# `episodios_por_punto` repite el análisis completo en cada punto (serie, climatología y episodios de calor y de frío) y los junta en una sola tabla.

# %%
episodios = episodios_por_punto(tabla_qc, tipos=["bahia", "pluma", "golfo", "laguna"])
resumen = episodios.groupby(["punto", "tipo"]).agg(episodios=("inicio", "size"), obs_max=("n_obs", "max"),
                                                  duracion_max=("duracion", "max"),
                                                  intensidad_max=("intensidad_max", lambda x: x.loc[x.abs().idxmax()])).round(2)
display(resumen)
print(f"Total: {len(episodios)} episodios")
activos = episodios[episodios["en_curso"]]               # episodios que siguen activos en la última observación
if not activos.empty:
    actual = activos.iloc[-1]
    print(f"Episodio en curso: {actual['nombre']} ({actual['tipo']}), desde el {actual['inicio'].date()}, "
          f"{actual['n_obs']} observaciones, anomalía máxima {actual['intensidad_max']:.2f} °C. Su duración es provisoria: puede crecer o terminar.")

# %% [markdown]
# Son pocos episodios y todos **cortos en observaciones**: la mayoría tiene 2 observaciones seguidas (duración mínima de 9 a 49 días). Con tan pocos datos solo vemos "huellas" de las anomalías. Veamos cuántos aparecen **a la vez en más de un punto**:

# %%
coincide = [int(((episodios["punto"] != fila.punto) & (episodios["tipo"] == fila.tipo) &
                 (episodios["inicio"] <= fila.fin) & (episodios["fin"] >= fila.inicio)).any()) for fila in episodios.itertuples()]
print(f"{sum(coincide)} de {len(episodios)} episodios coinciden en fechas con un episodio del mismo tipo en otro punto")
episodios[episodios["inicio"].between("2016-07-25", "2016-08-20")][["punto", "tipo", "inicio", "fin", "n_obs", "intensidad_max"]].round(2)

# %% [markdown]
# Cerca de la mitad coincide con otro punto. Eso puede ser un fenómeno **regional** real (una masa de aire cálido, una corriente) o un **artefacto** (una nube delgada o un error de la corrección atmosférica que cubrió toda la zona). Del 1 al 17 de agosto de 2016, por ejemplo, tres puntos distintos (la pluma, San Vicente y la laguna) tuvieron un episodio cálido a la vez. Landsat, con una sola pasada, no permite distinguirlos: sirve para formular la hipótesis, no para confirmarla.
#
# ### Tu turno 3: sensibilidad de los episodios
#
# Los parámetros de un episodio son convenciones, como los de Hobday. Cambia los valores de la celda siguiente (percentil 95, 3 observaciones seguidas, brecha de 16 o de 40 días) y observa cuánto cambia el conteo de episodios cálidos.
#
# * ¿Qué parámetro mueve más el conteo?
# * Si quisieras comparar tus resultados con los de otro estudio, ¿qué convendría dejar fijo?

# %%
percentil = 90          # prueba 95 (más exigente) u 85
min_obs = 2             # prueba 3 o 1
brecha_max_dias = 24    # prueba 16 o 40

st_golfo = tabla_qc[tabla_qc["id"] == "golfo_arauco"].set_index("fecha")["st_c"].astype(float)      # el punto con más episodios cálidos
clima_t = climatologia_ventana(st_golfo, percentiles=(percentil, 100 - percentil))
ev_t = detectar_episodios(st_golfo, clima_t, "calor", min_obs=min_obs, brecha_max_dias=brecha_max_dias, columna_umbral=f"p{percentil}")
print(f"Golfo de Arauco, p{percentil}, al menos {min_obs} observaciones, brecha de hasta {brecha_max_dias} días: {len(ev_t)} episodios cálidos "
      f"({int(ev_t['n_obs'].sum()) if len(ev_t) else 0} observaciones en total)")

# %% [markdown]
# ## 5. Lo que Landsat hace distinto: agua frente a tierra, y temperatura con reflectancia
#
# ### 5.1 Agua frente a tierra
#
# Cinco de los puntos son agua y uno es un **humedal** (tierra con vegetación). Landsat mide la **superficie**, así que el humedal se calienta mucho más que el agua. Comparemos la temperatura mediana, la amplitud estacional y las reflectancias calibradas de cada punto: el NDWI y el NDVI (calculados con la reflectancia, no con los DN) confirman qué es cada punto.

# %%
resumen_puntos = tabla_qc.groupby(["id", "tipo"]).agg(
    st_mediana=("st_c", "median"), st_p5=("st_c", lambda s: s.quantile(0.05)), st_p95=("st_c", lambda s: s.quantile(0.95)),
    ndwi=("ndwi", "median"), ndvi=("ndvi", "median"), rojo=("rojo", "median")).round(3)
resumen_puntos["amplitud_p95_p5"] = (resumen_puntos["st_p95"] - resumen_puntos["st_p5"]).round(1)
resumen_puntos.sort_values("st_mediana")

# %%
fig, ax = plt.subplots(figsize=(9, 3.8))
colores = {"bahia": AZUL, "pluma": "#d68910", "golfo": "#1e8449", "laguna": "#8e44ad", "humedal": ROJO}
for punto in tabla_qc["id"].unique():
    s = tabla_qc[tabla_qc["id"] == punto].set_index("fecha")["st_c"].dropna()
    tipo = tabla_qc.loc[tabla_qc["id"] == punto, "tipo"].iloc[0]
    mensual = s.groupby(s.index.month).median()
    ax.plot(mensual.index, mensual, "o-", ms=4, color=colores[tipo], label=f"{punto} ({tipo})")
ax.set_xticks(range(1, 13), labels=["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"])
ax.set_ylabel("mediana mensual (°C)")
ax.set_title("Ciclo anual de la temperatura de superficie: el humedal (tierra) frente al agua")
ax.legend(fontsize=7, ncol=2)
plt.show()

# %% [markdown]
# El humedal tiene una **mediana de unos 24 °C** frente a **13 a 19 °C** del agua, y un rango entre el percentil 5 y el 95 de unos **25 °C** frente a 5 a 13 °C: la tierra se calienta y enfría rápido, el agua tiene inercia térmica. Una anomalía de 2 °C en el agua es importante; en la tierra es ruido. Por eso cada punto usa **su propia climatología**: nunca comparamos un punto con otro, solo consigo mismo. Fíjate también en el NDWI y el NDVI calibrados: positivos y negativos en el agua, al revés en el humedal.
#
# ### 5.2 Temperatura y reflectancia en el mismo instante
#
# La pluma del Biobío y la bahía tienen aguas muy distintas: la pluma lleva sedimentos del río y su **reflectancia roja** es varias veces mayor. ¿Se relaciona la turbidez con la temperatura? Calculamos la **anomalía de ambas variables** (para quitar el ciclo estacional) y vemos si se mueven juntas.

# %%
def anomalia_de(variable, punto, tabla_origen=tabla_qc):
    s = tabla_origen[tabla_origen["id"] == punto].set_index("fecha")[variable].astype(float)
    return anomalias_observaciones(s, climatologia_ventana(s))["anomalia"].dropna()


fig, ejes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
filas = []
for ax, punto in zip(ejes, ["bahia_concepcion", "biobio_pluma", "golfo_arauco"]):
    a_st, a_rojo = anomalia_de("st_c", punto), anomalia_de("rojo", punto)
    comunes = a_st.index.intersection(a_rojo.index)
    r = np.corrcoef(a_st[comunes], a_rojo[comunes])[0, 1]
    ax.scatter(a_rojo[comunes] * 1000, a_st[comunes], s=10, color=AZUL)
    ax.set_title(f"{punto} (r = {r:.2f})", fontsize=9)
    ax.set_xlabel("anomalía de la reflectancia roja (x 1000)")
    filas.append({"punto": punto, "observaciones": len(comunes), "correlación": round(r, 2)})
ejes[0].set_ylabel("anomalía de temperatura (°C)")
plt.tight_layout()
plt.show()
pd.DataFrame(filas).set_index("punto")

# %% [markdown]
# Las correlaciones son **negativas y de débiles a moderadas** (de -0,2 a -0,4): las observaciones más brillantes tienden a ser más frías. Hay dos lecturas posibles y **los datos no permiten elegir**: (1) **residuo de nube o neblina**, que brilla y enfría la medida y que el control de calidad no eliminó del todo; (2) una **pluma o un día de agua más turbia y más fría** (por ejemplo, tras una lluvia de invierno). Sería necesario un dato más (caudal, viento, una imagen sin nubes del día) para decidir. Lo valioso es tener **las dos variables en la misma tabla**: la reflectancia sirvió para el control de calidad de la sección 2 y ahora para formular preguntas.

# %% [markdown]
# ## 6. Umbrales clásicos frente a aprendizaje automático
#
# Los umbrales (p90 y p10) son un método **basado en reglas**: tú decides el percentil, el número de observaciones y la brecha, y cada episodio se explica con una regla. La alternativa es dejar que un algoritmo de **aprendizaje automático** decida qué observaciones son "raras".
#
# **Isolation Forest** (bosque de aislamiento) es un método de detección de atípicos no supervisado. Construye muchos árboles de decisión con cortes aleatorios; los puntos raros quedan aislados con pocos cortes, los comunes necesitan muchos. No usa umbrales ni conoce el calendario: solo ve números. Por eso le damos **anomalías** (que ya no tienen estacionalidad) en vez de temperaturas.
#
# La novedad con Landsat: podemos darle **varias variables a la vez**. `aislamiento_forest_observaciones` le entrega a cada observación la anomalía de la temperatura, de la reflectancia roja y del NDWI. Hay que fijar la **contaminación**: la fracción de observaciones que se espera atípicas (aquí 10 %).

# %%
variables = ["st_c", "rojo", "ndwi"]
de_punto = tabla_qc[tabla_qc["id"] == PUNTO].set_index("fecha")
obs = de_punto[variables].astype(float)
clima_vars = {v: climatologia_ventana(obs[v]) for v in variables}

solo_temperatura = aislamiento_forest_observaciones(obs, ["st_c"], clima_vars, contaminacion=0.1)
multivariable = aislamiento_forest_observaciones(obs, variables, clima_vars, contaminacion=0.1)
print(f"Observaciones con las tres variables: {int(multivariable['puntaje'].notna().sum())}")
print(f"Atípicas con solo temperatura: {int(solo_temperatura['atipico'].sum())} | con las tres variables: {int(multivariable['atipico'].sum())}")
multivariable.nsmallest(5, "puntaje").round(3)

# %%
excede_calor = anomalias["excede_calor"]
excede_frio = anomalias["excede_frio"]
marca_umbral = (excede_calor | excede_frio)
comparacion = pd.DataFrame({
    "solo temperatura": solapamiento(marca_umbral, solo_temperatura["atipico"]),
    "tres variables": solapamiento(marca_umbral, multivariable["atipico"]),
}).T.rename(columns={"dias_a": "obs. sobre umbral", "dias_b": "obs. Isolation Forest", "comunes": "en común", "jaccard": "índice de Jaccard"})
comparacion.round(2)

# %% [markdown]
# El **índice de Jaccard** (observaciones en común / observaciones de la unión) va de 0 (ninguna en común) a 1 (idénticas). Con solo la temperatura comparten unas **dos de cada cinco** de las observaciones que marcan entre ambos; con las tres variables, **poco más de una de cada cinco**. Responden preguntas distintas.
#
# * El umbral pregunta: "¿esta temperatura, para esta época del año, supera lo que ocurre el 10 % de las veces?"
# * Isolation Forest pregunta: "¿esta combinación de anomalías es poco común en toda la serie?"
#
# Con las **tres variables**, el bosque también marca observaciones donde el **NDWI o la reflectancia** son raros aunque la temperatura sea casi normal. Contémoslas y veámoslas en el tiempo:

# %%
temperatura_normal = multivariable[multivariable["atipico"] & (multivariable["st_c"].abs() < 1)]
print(f"De {int(multivariable['atipico'].sum())} observaciones atípicas con tres variables, {len(temperatura_normal)} tienen una anomalía de temperatura menor que 1 °C")
temperatura_normal.round(3)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
a = multivariable["st_c"].dropna()
ax.plot(a.index, a, "o", ms=3, color="black", label="anomalía de temperatura")
raros = multivariable[multivariable["atipico"]]
ax.scatter(raros.index, raros["st_c"], s=40, facecolors="none", edgecolors="#f39c12", lw=1.5, label="atípico (Isolation Forest, 3 variables)")
marcados = anomalias[marca_umbral].dropna(subset=["valor"])
ax.scatter(marcados.index, marcados["anomalia"], s=12, color=ROJO, zorder=3, label="fuera de p10-p90")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("anomalía de temperatura (°C)")
ax.set_title("Umbrales clásicos (rojo) frente a Isolation Forest con tres variables (naranja)")
ax.legend(loc="lower left", fontsize=8)
plt.show()

# %% [markdown]
# ¿Y si le pedimos más? Con más contaminación el bosque marca más observaciones, pero el acuerdo con el umbral sube despacio (de 0,2 con 10 % a 0,3 con 30 %) a costa de marcar observaciones corrientes como "atípicas". La contaminación es un parámetro que **hay que elegir** y que no tiene una respuesta correcta desde los datos; en el umbral, el equivalente (el percentil 90) tiene una definición acordada por la comunidad.

# %%
filas = []
for contaminacion in (0.05, 0.1, 0.2, 0.3):
    f = aislamiento_forest_observaciones(obs, variables, clima_vars, contaminacion=contaminacion)
    filas.append({"contaminación": contaminacion, "observaciones atípicas": int(f["atipico"].sum()),
                  "Jaccard con el umbral": round(solapamiento(marca_umbral, f["atipico"])["jaccard"], 2)})
pd.DataFrame(filas).set_index("contaminación")

# %% [markdown]
# ### Umbrales clásicos frente a aprendizaje automático: qué significa cada uno
#
# | | Umbrales (p90 y p10 por día del año) | Isolation Forest (aprendizaje automático) |
# |---|---|---|
# | Qué detecta | Observaciones sobre el percentil 90 (o bajo el 10) de su época del año; episodios de 2 o más seguidas | Observaciones con combinaciones raras de anomalías, en toda la serie |
# | Estacionalidad | Integrada (un umbral por día del año) | No la conoce: hay que quitarla antes (anomalías) |
# | Duración | Regla explícita (observaciones seguidas, brecha máxima) | No tiene noción de episodio: marca observaciones sueltas |
# | Varias variables | Una a la vez | Sí: temperatura, reflectancia, índices a la vez |
# | Parámetros | Percentil, observaciones, brecha: convenciones del analista | Contaminación, árboles, variables: decisiones del analista |
# | Comparable entre estudios | Si se declaran los parámetros | Solo si comparten todas las decisiones |
# | Explicación | "Estuvo 2 veces seguidas sobre el p90" | "Su puntaje de aislamiento fue bajo" |
# | Cuándo conviene | Cuando se necesita comunicar un criterio simple | Para explorar y descubrir comportamientos raros, o con variables sin estándar |
#
# Ninguno es "mejor": responden preguntas distintas. Un buen flujo de trabajo usa los umbrales para **definir** episodios y Isolation Forest (u otros métodos) para **explorar** lo que esa definición no captura, como una observación rara en la reflectancia.

# %% [markdown]
# ### Una nota sobre Hobday y las olas de calor marinas
#
# El método de Hobday et al. (2016) es el estándar para **olas de calor marinas** (5 días seguidos sobre el p90 durante 30 años de base). Necesita una serie **diaria sin huecos** y larga, como los productos de temperatura del mar que interpolan satélites, boyas y barcos. **Landsat no sirve** para eso: pasa cada 8 días, las nubes borran muchas fechas y solo tiene 13 años. Por eso aquí trabajamos con **episodios** y no con "olas de calor marinas": usar el mismo nombre sería prometer más de lo que los datos permiten. `utils/extremos.py` conserva el método de Hobday (`climatologia_hobday`, `detectar_eventos`), validado contra la librería `xmhw`, por si tienes una serie diaria.

# %% [markdown]
# ## 7. Mapa de frecuencia de anomalías en una malla de 500 m
#
# Hasta ahora miramos puntos. Landsat da imágenes: la malla preparada (`landsat_malla_concepcion.nc`) tiene la **temperatura de superficie cada 500 m** en un cuadrado de 0,8 grados alrededor de la bahía de Concepción y el golfo de Arauco, para todas las fechas con al menos 10 % de píxeles válidos. Es una **muestra de un píxel cada 500 m** (sin promediar), con nubes como `NaN`.

# %%
with xr.open_dataset(DATOS / "landsat_malla_concepcion.nc") as ds:
    malla_bruta = ds["st_c"].load().astype("float32")      # cargar en memoria antes de cerrar el archivo
print(malla_bruta.sizes, "| fechas:", str(malla_bruta["time"].values[0])[:10], "a", str(malla_bruta["time"].values[-1])[:10])
print(f"Mínimo de la malla: {float(malla_bruta.min()):.1f} °C | máximo: {float(malla_bruta.max()):.1f} °C")

# %% [markdown]
# **Un mínimo de -43 °C no es una temperatura del suelo ni del agua de Concepción**: es justo el piso de la calibración (230 K), el valor de una nube muy fría que la máscara no marcó (las más frías ya quedaron como NaN). La malla (a diferencia de la tabla por punto) no tiene el control de calidad con la reflectancia, porque solo guarda la banda térmica. Hacemos un control **simple y declarado**: en esta zona, a esa hora (media mañana), ni el mar ni la tierra bajan de unos 3 °C, así que descartamos lo que está bajo ese valor.

# %%
muy_frio = (malla_bruta < 3).sum() / malla_bruta.notnull().sum()
print(f"{float(muy_frio):.1%} de los valores son menores que 3 °C: se descartan (casi todos nubes sin marcar)")
malla = malla_bruta.where(malla_bruta >= 3)
validas = malla.notnull().sum("time")
print(f"Observaciones válidas por píxel: de {int(validas.min())} a {int(validas.max())} (mediana {int(validas.median())})")

# %%
# Las coordenadas de los puntos están en grados (EPSG:4326); la malla está en UTM 18 S (EPSG:32718)
a_utm = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
puntos_xy = {fila.id: a_utm.transform(fila.lon, fila.lat) for fila in tabla.drop_duplicates("id").itertuples()}
verano = malla.sel(time=malla["time"].dt.month.isin([12, 1, 2])).mean("time")
x_km, y_km = malla["x"] / 1000, malla["y"] / 1000

fig, ejes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, datos_mapa, titulo, cmap in ((ejes[0], validas, "Observaciones válidas por píxel", "viridis"),
                                     (ejes[1], verano, "Temperatura media de verano (dic-feb, °C)", "RdYlBu_r")):
    imagen = ax.pcolormesh(x_km, y_km, datos_mapa, cmap=cmap, shading="nearest")
    for nombre, (px, py) in puntos_xy.items():
        ax.plot(px / 1000, py / 1000, "k^", ms=5, mec="white")
    ax.set_title(titulo)
    ax.set_xlabel("este (km, UTM 18 S)")
    ax.set_aspect("equal")
    plt.colorbar(imagen, ax=ax, shrink=0.75)
ejes[0].set_ylabel("norte (km)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se ve la **costa**: el mar (azul oscuro, a la izquierda) es mucho más frío en verano que la tierra (tonos claros). Las observaciones válidas cambian por zonas: los cortes rectos son los **bordes de las escenas** de Landsat (donde se juntan dos órbitas hay más pasos), y la zona oscura sobre la costa y los valles tiene muy pocas observaciones. En un píxel de esa zona casi no hay observaciones de noviembre a febrero: una hipótesis es que la máscara de nubes de Landsat toma por nube superficies muy brillantes y calientes (techos, arena, suelo desnudo). No lo comprobamos, pero cualquier promedio ahí estaría sesgado hacia los meses fríos. Los triángulos son los puntos del taller.
#
# ### Anomalías píxel por píxel
#
# `climatologia_malla` calcula la climatología por ventana de días **en cada píxel** (base 2013-2024), y `frecuencia_anomalias_malla` cuenta la fracción de observaciones por encima del p90 (o bajo el p10). Por construcción, en el periodo base esa fracción es cercana a **0,10** en todos los píxeles. Si miramos solo **los últimos años (2025 y 2026)**, una fracción mucho mayor que 0,10 indica que esos píxeles estuvieron más cálidos de lo normal.

# %%
clima_malla = climatologia_malla(malla, paso=16)             # un centro cada 16 días del año: la curva es suave y así tarda la mitad
frec_calor = frecuencia_anomalias_malla(malla, clima_malla, "calor", periodo=(2025, 2026))
frec_frio = frecuencia_anomalias_malla(malla, clima_malla, "frio", periodo=(2025, 2026))
print("Observaciones de 2025-2026 por píxel: mediana", int(frec_calor["n_obs"].median()))
print(f"Fracción sobre el p90 en 2025-2026: mediana de la malla {float(frec_calor['fraccion'].median()):.2f} (esperado 0,10)")
print(f"Fracción bajo el p10 en 2025-2026:  mediana de la malla {float(frec_frio['fraccion'].median()):.2f} (esperado 0,10)")

fig, ejes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, frec, titulo, cmap in ((ejes[0], frec_calor, "Fracción de observaciones sobre el p90", "Reds"),
                               (ejes[1], frec_frio, "Fracción de observaciones bajo el p10", "Blues")):
    imagen = ax.pcolormesh(x_km, y_km, frec["fraccion"], cmap=cmap, vmin=0, vmax=0.5, shading="nearest")
    ax.set_title(f"{titulo}, 2025-2026")
    ax.set_xlabel("este (km)")
    ax.set_aspect("equal")
    plt.colorbar(imagen, ax=ax, shrink=0.75)
ejes[0].set_ylabel("norte (km)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Cómo leerlo.**
#
# * Con unas **45 observaciones por píxel** en 2025-2026, la fracción tiene un error de unos 0,05 (la raíz de 0,1 x 0,9 / 45): un píxel con 0,2 no es necesariamente "más cálido que lo normal". Mira el patrón espacial (zonas grandes), no píxeles sueltos.
# * La mediana de la malla es de unos **0,15 para calor y 0,13 para frío**, por encima de 0,10 en los dos casos: 2025 y 2026 tuvieron **más observaciones extremas de los dos signos** que la climatología 2013-2024, no solo más calor.
# * Tampoco sabemos **por qué**. Un periodo así puede ser clima, nubes que se colaron, el cambio de sensor (Landsat 9 desde 2022) o una climatología base que no es homogénea (Tu turno 4). Es una **hipótesis**, no una demostración.
#
# ### Tu turno 4: otra mirada al mapa
#
# 1. Cambia el periodo a `(2013, 2018)` y a `(2019, 2024)`. Toda la base juntos da una fracción cercana a 0,10, pero cada mitad no tiene por qué: ¿cuál tiene más observaciones cálidas? ¿Qué te dice eso de qué tan "homogénea" es la climatología?
# 2. Mapea la **anomalía de la última escena despejada** con `anomalia_malla(malla, clima_malla)`: ¿se parece al patrón de frecuencia?
# 3. Para comunicar el resultado: ¿cuál de los dos mapas (cálido o frío) le mostrarías a una persona que trabaja en pesca artesanal de la zona y por qué?

# %%
# Punto de partida: la anomalía de la última escena con más del 50 % de píxeles válidos
anom = anomalia_malla(malla, clima_malla)
buena = (anom.notnull().mean(("x", "y")) > 0.5)
ultima = anom.sel(time=buena[buena].time[-1])
print("Última escena útil:", str(ultima["time"].values)[:10], "| anomalía mediana:", f"{float(ultima.median()):.2f} °C")

# %% [markdown]
# ## 8. Guardar los resultados
#
# Los episodios detectados son un **dato derivado**: si no los guardamos con su documentación, nadie (ni tú en tres meses) sabrá con qué parámetros salieron. Guardamos una tabla en parquet con los metadatos dentro del archivo, igual que hicimos con la serie de Sentinel-2. En los bloques 8 y 9 lo versionaremos y lo publicaremos.

# %%
import tempfile

SALIDAS = Path("/content/salidas") if "google.colab" in sys.modules else Path(tempfile.gettempdir()) / "salidas_taller"
SALIDAS.mkdir(parents=True, exist_ok=True)

ruta_episodios = SALIDAS / "episodios_landsat_concepcion.parquet"
guardar_series(episodios, ruta_episodios, metadatos={
    "titulo": "Episodios cálidos y fríos de la temperatura de superficie en el agua de la bahía de Concepción y alrededores (Landsat 8 y 9, 2013-2026)",
    "fuente": "USGS Landsat 8 y 9 Collection 2 Level-2 (banda térmica lwir11) leído desde Microsoft Planetary Computer; "
              "cinco puntos de agua (bahía, San Vicente, pluma del Biobío, golfo de Arauco, laguna Grande de San Pedro)",
    "licencia": "Dominio público (datos del USGS)",
    "periodo": f"{tabla['fecha'].min().date()} a {tabla['fecha'].max().date()}",
    "metodo": "Anomalías respecto de una climatología por ventana de días (utils/extremos.py); inspirado en Hobday et al. 2016, "
              "pero no es una ola de calor marina: Landsat no da una serie diaria",
    "parametros": {"base": "2013-2024", "ventana_dias": 31, "suavizado_dias": 15, "percentil_calor": 90, "percentil_frio": 10,
                   "observaciones_minimas_por_ventana": 8, "observaciones_seguidas_minimas": 2, "brecha_maxima_dias": 24,
                   "control_de_calidad": "agua con reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02: descartada"},
    "limitaciones": "Landsat pasa cada 8 días y las nubes borran muchas fechas (unas 10 a 25 observaciones válidas por año): "
                    "la duración de un episodio es una cota mínima y la tabla subestima la frecuencia real. La banda térmica "
                    "tiene 100 m nativos y mide la temperatura de la piel de la superficie (errores de 1 K o más): no es un producto oceanográfico.",
    "columnas": {"punto": "id del punto (puntos_concepcion.csv)", "nombre": "nombre del punto", "tipo": "calor o frio",
                 "inicio": "fecha de la primera observación del episodio", "fin": "fecha de la última observación",
                 "duracion": "días de la primera a la última observación, ambas incluidas (cota mínima)",
                 "n_obs": "observaciones válidas consecutivas del episodio", "dia_pico": "fecha de mayor anomalía",
                 "intensidad_max": "°C sobre la media climatológica (negativa en frío)", "intensidad_media": "°C",
                 "en_curso": "True si termina en la última observación válida de la serie"},
})
print(f"Guardado: {ruta_episodios}  ({ruta_episodios.stat().st_size / 1024:.0f} KB, {len(episodios)} episodios)")
print([k.decode() for k in pq.read_schema(ruta_episodios).metadata if k != b"pandas"])     # claves de los metadatos guardados

# %% [markdown]
# ## Resumen
#
# * **Landsat 8 y 9** miden temperatura de superficie y reflectancia en el mismo instante, pero llegan como **números enteros (DN)**: hay que **calibrarlos** (temperatura = DN x 0,00341802 + 149 K; reflectancia = DN x 0,0000275 - 0,2) y comprobar la calibración.
# * La serie es **irregular**: un paso cada 8 días, de los cuales solo unos 10 a 25 al año salen sin nubes, con menos dato en invierno (sesgo de cielo despejado) y huecos de más de 100 días.
# * La máscara de nubes **no ve todo**: la reflectancia (roja o SWIR1 altas sobre agua) delata las nubes que se colaron y que producen "agua muy fría" falsa. Descartarlas con una regla simple y declarada es parte del método.
# * No se puede aplicar tal cual el método de Hobday (necesita una serie diaria y 30 años). Se adapta la idea: **climatología con una ventana de 31 días**, umbrales p90 y p10, y **episodios** de 2 observaciones seguidas, sin cruzar huecos de más de 24 días. La duración es una cota mínima y la tabla **subestima** la frecuencia real; además el muestreo cambió en 2022 (Landsat 9), así que el conteo de episodios no sirve para ver tendencias.
# * Agua y tierra se separan con la reflectancia calibrada (NDWI, NDVI); la tierra tiene un ciclo anual de temperatura mucho más amplio, por eso cada punto usa su propia climatología.
# * **Isolation Forest** marca observaciones raras en toda la serie, y con Landsat puede usar **varias variables a la vez**. Comparte solo parte de lo que marca el umbral: son preguntas distintas.
# * En la malla de 500 m, la fracción de observaciones sobre el p90 muestra dónde hubo más anomalías cálidas que en la climatología 2013-2024; con unas 45 observaciones por píxel tiene ruido y hay que mirar patrones, no píxeles.
#
# ### Para el siguiente bloque
#
# En el bloque 8 refactorizaremos este código en funciones, lo versionaremos con Git y lo subiremos a GitHub. Lo que guardaste en esta sesión (`episodios_landsat_concepcion.parquet`) es un buen candidato para publicar en el bloque 9.
#
# **Referencias**
#
# * Earth Resources Observation and Science (EROS) Center (2020). Landsat 8-9 Operational Land Imager / Thermal Infrared Sensor Level-2, Collection 2 [dataset]. *U.S. Geological Survey*. https://doi.org/10.5066/P9OGBGM6
# * Malakar, N. K., Hulley, G. C., Hook, S. J., Laraby, K., Cook, M. y Schott, J. R. (2018). An operational land surface temperature product for Landsat thermal data: methodology and validation. *IEEE Transactions on Geoscience and Remote Sensing*, 56(10), 5717-5735. https://doi.org/10.1109/TGRS.2018.2824828
# * Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227-238. https://doi.org/10.1016/j.pocean.2015.12.014
# * Liu, F. T., Ting, K. M. y Zhou, Z.-H. (2008). Isolation Forest. *Proceedings of the 8th IEEE International Conference on Data Mining*, 413-422. https://doi.org/10.1109/ICDM.2008.17
