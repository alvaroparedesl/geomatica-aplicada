# %% [markdown]
# # Bloque 6. Visualización interactiva e interpretación
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/06_visualizacion_interactiva.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# Taller *Datos FAIR, series temporales satelitales y detección de anomalías con Python*. Día 2, mañana (unos 50 minutos).
#
# Un gráfico estático responde la pregunta que le hiciste al hacerlo. Uno interactivo te deja hacerle preguntas nuevas: acercarte a un mes, ver el valor exacto de una fecha, esconder un punto. Pero antes de hacer interactivo un gráfico conviene que sea un **buen gráfico**.
#
# **Qué vas a hacer**
#
# | Parte | Tema | Minutos |
# |---|---|---|
# | 1 | Buenas prácticas para gráficos de series ambientales | 15 |
# | 2 | Series interactivas con `hvplot` | 20 |
# | 3 | Mapa de los puntos con `folium` y resumen de cada serie | 10 |
# | 4 | Ejercicio de interpretación | 5 |

# %% [markdown]
# ## Preparar el entorno
#
# La celda `Preparar entorno` clona el repositorio del curso (para usar `utils/`). **`hvplot` no viene en Colab**, así que la celda siguiente lo instala, solo si falta (usa `bokeh`, `holoviews` y `panel`, que Colab ya trae). Puede tardar un minuto.

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
# @title Instalar hvplot (solo si falta)
import importlib.util

if importlib.util.find_spec("hvplot") is None:
    requisitos = [linea.strip() for linea in (RAIZ / "talleres/fair-series-temporales/requirements.txt").read_text().splitlines()
                  if linea.strip() and not linea.startswith("#")]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[r for r in requisitos if r.split()[0] == "hvplot"]], check=True)
    print("hvplot instalado")
else:
    print("hvplot ya está instalado")

# %%
# @title Importar bibliotecas
import base64
import io
import tempfile

import folium
import hvplot
import hvplot.pandas  # noqa: F401  (activa el accesor .hvplot de pandas)
import matplotlib.pyplot as plt
import pandas as pd

# Funciones de apoyo del curso (carpeta utils/)
from utils.series import cargar_puntos_csv, cargar_series, rolling_media, serie_de_punto
from utils.viz import mapa_puntos, serie_interactiva

hvplot.extension("bokeh")  # el motor de gráficos interactivos que usamos

# Una paleta que se distingue bien con daltonismo (Okabe-Ito), un color por punto
PALETA = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# %% [markdown]
# Los datos: la serie de Sentinel-2 de 12 meses (la misma de los bloques 4 y 5) y los puntos.

# %%
tabla = cargar_series("series_sentinel2_concepcion.parquet")
puntos = cargar_puntos_csv("puntos_concepcion.csv")
colores = dict(zip(puntos["id"], PALETA))  # el mismo color para el mismo punto en todos los gráficos
print(f"{len(tabla)} filas, {tabla['id'].nunique()} puntos, {tabla['fecha'].min():%Y-%m-%d} a {tabla['fecha'].max():%Y-%m-%d}")

# %% [markdown]
# ## 1. Buenas prácticas para gráficos de series ambientales
#
# Mira primero un gráfico **mejorable** de los mismos datos (reflectancia roja de dos puntos), y después el corregido. Los dos usan exactamente la misma tabla.

# %%
dos = ["biobio_pluma", "bahia_concepcion"]
fig, eje = plt.subplots(figsize=(10, 3.4))
for punto, color in zip(dos, ["red", "green"]):
    s = tabla[tabla["id"] == punto].set_index("fecha")["b04"].ffill()  # rellena los vacíos con el último valor
    eje.plot(s.index, s.values, color=color, linewidth=2.5)
eje.set_ylim(0, 0.35)
eje.set_title("B04")
plt.tight_layout()
plt.show()

# %% [markdown]
# ¿Qué le pasa a este gráfico?
#
# * No dice **qué punto es cada línea** (no tiene leyenda), ni las **unidades** de los ejes.
# * Rojo y verde no se distinguen con daltonismo.
# * Rellenó con el último valor los días sin dato (`ffill`): parece que hubo medición todos los días, y **inventa** tramos planos.
# * Une todo con una línea gruesa sin marcar las **observaciones**: no se ve que las fechas son irregulares.
# * El eje Y está fijado en 0,35 por el valor más extremo de otro punto, y aplasta a los demás.
#
# Ahora el mismo dato con buenas prácticas:

# %%
fig, eje = plt.subplots(figsize=(10, 3.6))
for punto in dos:
    s = tabla[tabla["id"] == punto].dropna(subset=["b04"])  # sin rellenar: solo las fechas con dato
    eje.plot(s["fecha"], s["b04"], marker="o", markersize=3.5, linewidth=0.9, color=colores[punto], label=punto.replace("_", " "))
eje.set_ylabel("Reflectancia roja B04 (adimensional, 0 a 1)")
eje.set_xlabel("Fecha de la toma (UTC)")
eje.set_title("Turbidez: la pluma del Biobío frente a la bahía de Concepción")
eje.legend(frameon=False)
eje.grid(alpha=0.3)
fig.text(0.01, 0.01, "Fuente: Sentinel-2 L2A (Copernicus, vía Planetary Computer). Mediana en una ventana de 150 m. Los marcadores son tomas.",
         fontsize=7, color="0.4")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# %% [markdown]
# **Lista para tus gráficos de series ambientales**
#
# 1. **Cada eje con nombre y unidades.** Un título que diga qué se ve, no solo la variable.
# 2. **Marca las observaciones** (marcadores) cuando las fechas son irregulares, y **no unas ni rellenes los vacíos** sin decirlo: un vacío es información.
# 3. **Un color por punto, el mismo en todo el cuaderno**, y una paleta apta para daltonismo (Okabe-Ito, ColorBrewer). Etiqueta las líneas directamente o con leyenda.
# 4. **Eje Y honesto.** Si comparas puntos, elige la misma escala o dilo. No cortes el eje de forma que exagere un cambio pequeño.
# 5. **Muestra la incertidumbre y la calidad**: cuántos píxeles válidos hay detrás, o qué fechas tienen pocos datos.
# 6. **Un gráfico, una idea.** Con varias variables, paneles pequeños con el eje X compartido antes que dos ejes Y.
# 7. **Cita la fuente** y el procesamiento (aquí: Sentinel-2, mediana en una ventana de 150 m).
#
# **Tu turno 1.** Pasa el gráfico de arriba a paneles pequeños (uno por punto) con `plt.subplots(3, 2, sharex=True, sharey=True)`. ¿Qué se gana y qué se pierde frente al gráfico con todos juntos?

# %%
# Tu turno 1
# fig, ejes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey=True)
# for eje, (punto, datos) in zip(ejes.ravel(), tabla.groupby("id", sort=False)):
#     ...

# %% [markdown]
# ## 2. Series interactivas con `hvplot`
#
# [`hvplot`](https://hvplot.holoviz.org/) agrega a `pandas` un método `.hvplot` casi igual a `.plot`, pero el resultado es un gráfico de Bokeh: **zoom con la rueda del mouse, arrastrar para desplazarse, valores exactos al pasar el cursor y clic en la leyenda para esconder una serie**. Los controles aparecen arriba a la derecha del gráfico.
#
# ### 2.1 Con el helper del curso
#
# `serie_interactiva` arma por ti el gráfico: una línea con marcadores por punto, una por cada valor de `id`. Las fechas sin valor dejan huecos.

# %%
serie_interactiva(tabla, "ndwi", titulo="NDWI por punto")

# %% [markdown]
# Pruébalo: acércate a un mes con la herramienta de zoom, esconde con un clic en la leyenda los puntos que no te interesan, y busca el valor del humedal (NDWI muy negativo, porque es tierra).
#
# ### 2.2 La misma idea con la llamada de `hvplot`, paso a paso
#
# El helper solo prepara estos argumentos. La llamada de la biblioteca es esta, con cada parámetro a la vista:

# %%
grafico = tabla.hvplot.line(
    x="fecha",  # columna del eje X
    y="b04",  # columna del eje Y
    by="id",  # una línea por cada valor de "id"
    width=900, height=380,
    ylabel="Reflectancia roja B04", xlabel="Fecha",
    title="B04 por punto",
    legend="right",
    hover_cols=["nubes_escena", "pixeles_validos"],  # datos de calidad en el cuadro que aparece al pasar el cursor
)
grafico

# %% [markdown]
# Pasa el cursor por un punto: además del valor, el cuadro muestra las nubes de la escena y la fracción de la ventana válida. Poner **la calidad del dato en el tooltip** es una de las mejores ventajas de un gráfico interactivo.
#
# Los gráficos de `hvplot` se **combinan** con `*` (uno sobre otro) y `+` (uno junto a otro). Por ejemplo, las observaciones de la pluma del Biobío con su media móvil de 30 días (`rolling_media`, del bloque 5):

# %%
pluma = serie_de_punto(tabla, "biobio_pluma", "b04")
df_pluma = pd.DataFrame({"b04": pluma, "media móvil 30 d": rolling_media(pluma, "30D", minimo=3)}).reset_index()

observaciones = df_pluma.hvplot.scatter(x="fecha", y="b04", color="gray", size=30, label="observaciones")
suavizada = df_pluma.dropna(subset=["media móvil 30 d"]).hvplot.line(x="fecha", y="media móvil 30 d", color="#0072B2", line_width=3, label="media móvil 30 d")
(observaciones * suavizada).opts(width=900, height=350, title="Pluma del Biobío: turbidez y media móvil", ylabel="B04", legend_position="top_left")

# %% [markdown]
# Y varios gráficos en una cuadrícula, con `subplots=True` y `.cols(2)`, uno por punto (como el ejercicio de las buenas prácticas, pero con zoom):

# %%
tabla.hvplot.line(x="fecha", y="ndci", by="id", subplots=True, width=430, height=230, ylabel="NDCI", xlabel="", shared_axes=True).cols(2)

# %% [markdown]
# Con `groupby="id"` en vez de `by="id"` obtienes **un solo gráfico con un menú desplegable** para elegir el punto. Ese menú es un widget de `panel`: en Colab a veces se comporta distinto que en Jupyter, así que si no aparece o no responde, no te quedes pegado: usa `by` o `subplots`.

# %%
tabla.hvplot.line(x="fecha", y="ndvi", groupby="id", width=800, height=330, ylabel="NDVI", title="NDVI (elige el punto)")

# %% [markdown]
# Un gráfico se puede guardar como un archivo HTML **independiente**, que cualquiera abre en el navegador sin Python. Es una buena forma de compartir un resultado (y de agregarlo a tu repositorio y a tu depósito en el bloque 9).

# %%
ruta_html = Path(tempfile.gettempdir()) / "b04_por_punto.html"
hvplot.save(grafico, str(ruta_html))
print(f"Guardado: {ruta_html} ({ruta_html.stat().st_size / 1024:.0f} kB)")

# %% [markdown]
# **Tu turno 2.**
#
# 1. Con `serie_interactiva`, grafica `"ndci"` solo para las bahías y el golfo (`puntos=["bahia_concepcion", "san_vicente", "golfo_arauco"]`).
# 2. Con `tabla.hvplot.line(...)`, grafica `"ndwi"` de la laguna y el golfo, con `pixeles_validos` en el tooltip.
# 3. Combina con `*` la serie de B04 de la bahía con su media móvil de 60 días.

# %%
# Tu turno 2

# %% [markdown]
# ## 3. Mapa de los puntos con `folium`
#
# [`folium`](https://python-visualization.github.io/folium/) dibuja mapas interactivos (Leaflet) a partir de Python. Ya lo usaste en el bloque 4 con el helper `mapa_puntos`: un color por tipo de punto, una capa por tipo que se enciende y se apaga, y el círculo de 150 m de la ventana.

# %%
mapa_puntos(puntos, radio_m=150)

# %% [markdown]
# Hagamos ahora un mapa donde el **cuadro emergente (popup)** de cada punto resuma su serie: cuántas fechas válidas tiene, sus estadísticos y un gráfico pequeño de la serie. Es folium "a mano": una función que arma el HTML del popup y un bucle que agrega un marcador por punto.
#
# Primero, el resumen y el gráfico pequeño (una imagen PNG incrustada en el HTML):

# %%
def resumen_html(id_punto):
    """Tabla HTML con las estadísticas de las cuatro variables de un punto."""
    sub = tabla[tabla["id"] == id_punto]
    filas = ""
    for variable in ["ndwi", "ndci", "b04", "ndvi"]:
        v = sub[variable].dropna()
        filas += f"<tr><td>{variable.upper()}</td><td>{v.median():.3f}</td><td>{v.quantile(0.1):.3f}</td><td>{v.quantile(0.9):.3f}</td></tr>"
    return (f"<table style='font-size:12px'><tr><th>variable</th><th>mediana</th><th>p10</th><th>p90</th></tr>{filas}</table>"
            f"<p style='font-size:12px'>{sub['ndwi'].notna().sum()} fechas válidas de {len(sub)} con imagen</p>")


def miniatura_png(id_punto, variable):
    """Gráfico pequeño de una serie, como imagen PNG en base64 (para incrustar en el popup)."""
    s = serie_de_punto(tabla, id_punto, variable)
    fig, eje = plt.subplots(figsize=(3.4, 1.4))
    eje.plot(s.index, s.values, marker="o", markersize=2, linewidth=0.8, color=colores[id_punto])
    eje.set_title(f"{variable.upper()} en el tiempo", fontsize=8)
    eje.tick_params(labelsize=6)
    eje.grid(alpha=0.3)
    memoria = io.BytesIO()
    fig.savefig(memoria, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(memoria.getvalue()).decode()


print(resumen_html("humedal_rocuant"))

# %% [markdown]
# Y el mapa. Cada punto es un `folium.CircleMarker`; el popup es un `folium.Popup` con el HTML. Variable de la miniatura: NDVI para el humedal y NDWI para el resto.

# %%
mapa = folium.Map(location=[puntos["lat"].mean(), puntos["lon"].mean()], zoom_start=10, tiles=None)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mapa)
folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri, Maxar, Earthstar Geographics", name="Imagen satelital (Esri)").add_to(mapa)

for fila in puntos.itertuples():
    variable = "ndvi" if fila.tipo == "humedal" else "ndwi"
    html = (f"<b>{fila.nombre}</b><br>tipo: {fila.tipo}<br>{resumen_html(fila.id)}"
            f"<img src='data:image/png;base64,{miniatura_png(fila.id, variable)}' width='290'>")
    folium.CircleMarker(
        [fila.lat, fila.lon], radius=8, color="white", weight=2, fill=True, fill_color=colores[fila.id], fill_opacity=0.95,
        tooltip=fila.nombre, popup=folium.Popup(folium.IFrame(html, width=330, height=330), max_width=340),
    ).add_to(mapa)
    folium.Circle([fila.lat, fila.lon], radius=150, color=colores[fila.id], weight=1, fill=False).add_to(mapa)

folium.LayerControl().add_to(mapa)
mapa.fit_bounds([[puntos["lat"].min(), puntos["lon"].min()], [puntos["lat"].max(), puntos["lon"].max()]])
mapa

# %% [markdown]
# Haz clic en cada punto. Pasa de un lado a otro de la zona y compara: el popup de la pluma del Biobío frente al del golfo de Arauco, el humedal frente a la laguna. Cambia la capa base a la imagen satelital para ver qué hay alrededor.
#
# **Tu turno 3.** Agrega NDCI a la tabla del popup, o cambia el color del marcador según la mediana de NDCI (por ejemplo con `branca.colormap`, que viene con folium). Guarda el mapa como HTML con `mapa.save("mapa_puntos.html")`.

# %%
# Tu turno 3

# %% [markdown]
# ## 4. Ejercicio de interpretación
#
# Interpretar es pasar de "el gráfico sube" a "esto puede deberse a...". Usa los gráficos interactivos para responder, **por escrito** y en pareja:
#
# 1. **Pluma del Biobío (B04).** ¿En qué meses sube la reflectancia roja? ¿Cómo se compara con el golfo de Arauco y la bahía? ¿Qué datos externos usarías para ver si se debe a las lluvias o al caudal del río?
# 2. **Una fecha aislada.** Busca el valor más alto de B04 de todo el año (pista: la herramienta de zoom y el tooltip). ¿Cuántas nubes tenía la escena? ¿Cuántos píxeles de la ventana eran válidos? ¿Lo considerarías un evento real o un dato dudoso? ¿Qué mirarías para decidir?
# 3. **Clorofila (NDCI).** ¿Se ve un ciclo estacional en las bahías? ¿Qué cuidado debes tener en los meses con pocas fechas?
# 4. **Humedal (NDVI).** ¿Cuánto varía durante el año? ¿Qué ventaja tiene una ventana pequeña sobre un punto para esta cobertura?
#
# Un gráfico para empezar: B04 de los tres puntos de la pregunta 1, con la calidad en el tooltip.

# %%
tres = tabla[tabla["id"].isin(["biobio_pluma", "bahia_concepcion", "golfo_arauco"])]
tres.hvplot.scatter(x="fecha", y="b04", by="id", width=900, height=380, size=35, alpha=0.8, ylabel="B04",
                    hover_cols=["nubes_escena", "pixeles_validos"], title="B04: pluma, bahía y golfo")

# %% [markdown]
# <details>
# <summary>Pistas para la interpretación (ábrelas después de intentarlo)</summary>
#
# 1. En la serie, la pluma sube en noviembre y diciembre y en agosto y septiembre, y en otoño es parecida a las demás. Eso es compatible con más aporte del río, pero hay otras explicaciones (viento, reflejo del sol). Para confirmar harían falta lluvia y caudal de estaciones de la Dirección General de Aguas o de la Dirección Meteorológica.
# 2. El valor más alto es del golfo de Arauco, en enero, con una escena casi sin nubes y la ventana completa válida. No lo explica la máscara: hay que mirar la imagen (espuma, un barco, un reflejo) antes de llamarlo evento.
# 3. En invierno el NDCI de las bahías baja mucho, pero hay pocas fechas y el agua es muy oscura, así que la razón es inestable: no es una conclusión.
# 4. El NDVI del humedal varía poco (entre 0,5 y 0,6). Una ventana de 300 m resume muchos píxeles y evita que una sola celda con sombra o con agua distorsione el valor.
# </details>
#
# **Tu turno 4.** Escribe tres frases sobre el punto y la variable que más te llamaron la atención: lo que ves, tu hipótesis y el dato que necesitarías para confirmarla. Guárdalas en una celda de texto del cuaderno: son la base de la presentación del día 3.

# %% [markdown]
# ## Cierre
#
# * Un buen gráfico estático primero, luego interactivo: ejes con nombre y unidades, vacíos a la vista, color consistente.
# * `hvplot` da zoom, tooltip y filtrado casi sin código; pon la **calidad del dato** (nubes, píxeles válidos) en el tooltip.
# * `folium` hace un mapa donde cada punto cuenta su propia historia en un popup.
# * Una interpretación es una **hipótesis**: dice qué dato faltaría para confirmarla.
#
# En el bloque 7 usamos la serie larga de temperatura superficial del mar para detectar eventos extremos.
