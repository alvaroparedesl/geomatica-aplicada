"""
Funciones de visualización: composiciones RGB, paneles, histogramas, series
temporales y mapas interactivos.

Las funciones que dibujan tienen el parámetro `mostrar`: con True (por defecto)
muestran la figura; con False la devuelven, por si quieres guardarla o modificarla.
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .raster import extraer_series_puntos

# Nombre, longitud de onda y color de cada banda (Sentinel-2 y Landsat)
INFO_BANDAS = {
    "coastal": ("Aerosol costero", "443 nm", "lightskyblue"),
    "B02": ("Azul", "490 nm", "blue"),
    "blue": ("Azul", "482 nm", "blue"),
    "B03": ("Verde", "560 nm", "green"),
    "green": ("Verde", "562 nm", "green"),
    "B04": ("Rojo", "665 nm", "red"),
    "red": ("Rojo", "655 nm", "red"),
    "B05": ("Red Edge 1", "705 nm", "orange"),
    "B08": ("NIR", "842 nm", "darkred"),
    "nir08": ("NIR", "865 nm", "darkred"),
    "B11": ("SWIR 1", "1610 nm", "purple"),
    "swir16": ("SWIR 1", "1609 nm", "gold"),
    "B12": ("SWIR 2", "2190 nm", "brown"),
    "swir22": ("SWIR 2", "2201 nm", "orange"),
}

COLORES_PUNTOS = ["green", "red", "blue", "orange", "purple", "cyan"]


def _finalizar(fig, mostrar):
    """Figura de plotly: la muestra o la devuelve, según `mostrar`."""
    if mostrar:
        fig.show()
        return None
    return fig


def _finalizar_mpl(fig, mostrar):
    """Figura de matplotlib: la muestra o la devuelve, según `mostrar`."""
    if mostrar:
        plt.show()
        return None
    return fig


# ---------------------------------------------------------------------------
# Imágenes y paneles (matplotlib)
# ---------------------------------------------------------------------------
def crear_imagen_rgb(dataset, bandas_rgb, factor_brillo=3.0):
    """
    Arma una imagen de 3 bandas lista para `plt.imshow`.

    Sirve para color verdadero (rojo, verde, azul) o falso color (por ejemplo
    NIR, rojo, verde). Los valores se multiplican por `factor_brillo` y se
    recortan entre 0 y 1.

    Args:
        dataset (xarray.Dataset): Reflectancia de una sola fecha (sin dimensión time).
        bandas_rgb (list): Tres nombres de banda, en el orden [R, G, B] de la pantalla.
        factor_brillo (float): Aumenta el brillo (3 a 4 suele verse bien).

    Returns:
        numpy.ndarray: Arreglo (alto, ancho, 3) con valores entre 0 y 1.
    """
    rgb = np.stack([dataset[banda].values for banda in bandas_rgb], axis=-1)
    return np.clip(rgb * factor_brillo, 0, 1)


def mostrar_imagen(
    datos, titulo="", figsize=(15, 10), cmap=None, vmin=None, vmax=None, barra=None, mostrar=True
):
    """
    Muestra una imagen (RGB o una sola banda/índice) sin ejes.

    Args:
        datos: Arreglo o DataArray, 2D (con `cmap`) o (alto, ancho, 3) para RGB.
        titulo (str): Título de la figura.
        figsize (tuple): Tamaño de la figura.
        cmap (str, optional): Paleta de colores, por ejemplo "RdYlGn".
        vmin, vmax (float, optional): Límites de la escala de colores.
        barra (str, optional): Si se entrega, agrega barra de colores con esa etiqueta.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    fig, ax = plt.subplots(figsize=figsize)
    imagen = ax.imshow(datos, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=14)
    ax.axis("off")
    if barra is not None:
        fig.colorbar(imagen, ax=ax, fraction=0.046, label=barra)
    fig.tight_layout()
    return _finalizar_mpl(fig, mostrar)


def mostrar_paneles(paneles, ncols=2, titulo=None, mostrar=True):
    """
    Muestra varias imágenes en una cuadrícula, cada una con su título y paleta.

    Args:
        paneles (list): Lista de tuplas `(titulo, datos)` o `(titulo, datos, opciones)`.
            `opciones` es un diccionario con cmap, vmin, vmax y barra (True/False).
            Por defecto las imágenes 2D llevan barra de colores y las RGB no.
        ncols (int): Número de columnas.
        titulo (str, optional): Título general.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.

    Ejemplo:
        mostrar_paneles([("NDVI", ndvi, {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 1}),
                         ("RGB", rgb)])
    """
    nrows = -(-len(paneles) // ncols)
    fig, ejes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)

    for eje, panel in zip(ejes.ravel(), paneles):
        nombre, datos, *resto = panel
        opciones = dict(resto[0]) if resto else {}
        con_barra = opciones.pop("barra", np.ndim(datos) == 2)
        imagen = eje.imshow(datos, **opciones)
        eje.set_title(nombre)
        eje.axis("off")
        if con_barra:
            fig.colorbar(imagen, ax=eje, fraction=0.046)

    for eje in ejes.ravel()[len(paneles):]:
        eje.set_visible(False)

    if titulo:
        fig.suptitle(titulo, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _finalizar_mpl(fig, mostrar)


def mostrar_mascaras(mascaras, ncols=2, mostrar=True):
    """
    Muestra varias máscaras booleanas indicando cuántos píxeles son True.

    Args:
        mascaras (dict): {nombre: máscara (DataArray booleano 2D)}.
        ncols (int): Número de columnas.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    paneles = [
        (f"{nombre}\n({int(mascara.sum()):,} píxeles)",
         mascara.astype(int), {"cmap": "RdYlBu_r", "vmin": 0, "vmax": 1})
        for nombre, mascara in mascaras.items()
    ]
    return mostrar_paneles(paneles, ncols=ncols, titulo="Máscaras", mostrar=mostrar)


def graficar_capas(capas, titulo="", figsize=(10, 8), mostrar=True):
    """
    Dibuja varias capas vectoriales una encima de otra (ejes iguales y cuadrícula).

    Args:
        capas (list): Lista de tuplas `(gdf, color)` o `(gdf, color, transparencia)`.
        titulo (str): Título del gráfico.
        figsize (tuple): Tamaño de la figura.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.

    Ejemplo:
        graficar_capas([(poligono_a, "red"), (poligono_b, "blue")], "A y B")
    """
    fig, ax = plt.subplots(figsize=figsize)
    for capa in capas:
        gdf, color, *alpha = capa
        gdf.plot(ax=ax, color=color, alpha=alpha[0] if alpha else 0.5)
    ax.set_title(titulo)
    ax.set_aspect("equal")
    ax.grid(True)
    return _finalizar_mpl(fig, mostrar)


def graficar_histogramas(dataset, bandas, titulo="Histogramas por banda", bins=50, columnas=3, mostrar=True):
    """
    Un histograma por banda, con la media y la mediana marcadas.

    Args:
        dataset (xarray.Dataset): Reflectancia (sin dimensión time).
        bandas (list): Nombres de banda.
        titulo (str): Título general.
        bins (int): Número de intervalos.
        columnas (int): Histogramas por fila.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    filas = -(-len(bandas) // columnas)
    fig, ejes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), squeeze=False)

    for eje, banda in zip(ejes.ravel(), bandas):
        valores = dataset[banda].values.ravel()
        valores = valores[~np.isnan(valores)]
        nombre, longitud, color = INFO_BANDAS.get(banda, (banda, "", "gray"))
        media, mediana = valores.mean(), np.median(valores)

        eje.hist(valores, bins=bins, alpha=0.7, color=color, edgecolor="black")
        eje.axvline(media, color="red", linestyle="--", label=f"Media: {media:.3f}")
        eje.axvline(mediana, color="blue", linestyle="--", label=f"Mediana: {mediana:.3f}")
        eje.set_title(f"{banda} - {nombre}\n({longitud})")
        eje.set_xlabel("Reflectancia")
        eje.set_ylabel("Frecuencia")
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8)

    for eje in ejes.ravel()[len(bandas):]:
        eje.set_visible(False)

    fig.suptitle(titulo, fontsize=16)
    fig.tight_layout()
    return _finalizar_mpl(fig, mostrar)


# ---------------------------------------------------------------------------
# Gráficos interactivos (plotly)
# ---------------------------------------------------------------------------
def histograma_interactivo(dataset, bandas, titulo="Histogramas", rango=(0, 0.5), bins=100, mostrar=True):
    """
    Histogramas de varias bandas superpuestos en un solo gráfico interactivo.

    Args:
        dataset (xarray.Dataset): Reflectancia (sin dimensión time).
        bandas (list): Nombres de banda.
        titulo (str): Título del gráfico.
        rango (tuple): Rango de reflectancia a graficar.
        bins (int): Número de intervalos.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    fig = go.Figure()
    for banda in bandas:
        valores = dataset[banda].values.ravel()
        valores = valores[~np.isnan(valores)]
        frecuencia, bordes = np.histogram(valores, bins=bins, range=rango)
        color = INFO_BANDAS.get(banda, (None, None, "gray"))[2]
        fig.add_trace(
            go.Scatter(x=bordes[:-1], y=frecuencia, mode="lines", name=banda, line=dict(color=color, width=2))
        )
    fig.update_layout(title=titulo, xaxis_title="Reflectancia", yaxis_title="Frecuencia", width=800, height=500)
    return _finalizar(fig, mostrar)


def graficar_series_temporales(
    serie,
    columna_id="punto_id",
    indices=None,
    orientacion="filas",
    promediar=False,
    colores=None,
    titulo="Series temporales",
    mostrar=True,
):
    """
    Grafica la evolución en el tiempo de uno o más índices, una línea por punto.

    Args:
        serie (pandas.DataFrame): Resultado de `utils.raster.extraer_series_puntos`
            (columnas time, indice, valor y `columna_id`).
        columna_id (str): Columna que distingue las líneas (punto, tipo de uso...).
        indices (list, optional): Índices a graficar. Por defecto todos.
        orientacion (str): "filas" (un gráfico bajo otro) o "columnas" (lado a lado).
        promediar (bool): Si es True, promedia los puntos que comparten `columna_id`
            (por ejemplo varios puntos del mismo tipo de uso).
        colores (list, optional): Colores de las líneas.
        titulo (str): Título del gráfico.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    indices = list(indices or serie["indice"].unique())
    colores = colores or COLORES_PUNTOS
    en_filas = orientacion == "filas"

    fig = make_subplots(
        rows=len(indices) if en_filas else 1,
        cols=1 if en_filas else len(indices),
        subplot_titles=indices,
        shared_xaxes=en_filas,
    )
    for j, indice in enumerate(indices, start=1):
        fila, columna = (j, 1) if en_filas else (1, j)
        datos_indice = serie[serie["indice"] == indice].dropna(subset=["valor"])
        for i, (nombre, datos) in enumerate(datos_indice.groupby(columna_id, sort=False)):
            if promediar:
                datos = datos.groupby("time")["valor"].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=datos["time"],
                    y=datos["valor"],
                    mode="lines+markers",
                    name=str(nombre),
                    line=dict(color=colores[i % len(colores)], width=2),
                    marker=dict(size=4),
                    showlegend=(j == 1),
                    legendgroup=str(nombre),
                ),
                row=fila,
                col=columna,
            )
        fig.update_yaxes(title_text=indice, row=fila, col=columna)
        fig.update_xaxes(title_text="Fecha", row=len(indices) if en_filas else 1, col=columna)

    fig.update_layout(
        title=titulo,
        height=300 * len(indices) if en_filas else 600,
        hovermode="x unified",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return _finalizar(fig, mostrar)


def graficar_series_puntos(datos, coordenadas, variables=("NDVI",), mostrar=True):
    """
    Atajo: serie temporal de una o más variables en puntos dados por coordenadas.

    Args:
        datos (xarray.Dataset): Variables con dimensiones (time, y, x).
        coordenadas (list): Lista de [x, y] en el CRS de `datos` (por ejemplo UTM).
        variables (iterable): Variables a graficar (deben existir en `datos`).
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    puntos = pd.DataFrame(coordenadas, columns=["x", "y"])
    puntos["punto_id"] = [f"Punto {i + 1}" for i in range(len(puntos))]
    serie = extraer_series_puntos(datos[list(variables)], puntos)
    return graficar_series_temporales(serie, indices=variables, mostrar=mostrar)


def mostrar_imagen_interactiva(datos, modo="RGB", indice_tiempo=0, mostrar=True):
    """
    Imagen interactiva (zoom y valores al pasar el mouse) de una fecha del Dataset.

    Args:
        datos (xarray.Dataset): Reflectancia con dimensión time. Para los modos
            "NDVI" y "EVI" debe incluir esas variables (ver `calcular_indices`).
            Las bandas deben llamarse como en Landsat (red, green, blue, nir08).
        modo (str): "RGB", "Falso color", "NDVI" o "EVI".
        indice_tiempo (int): Posición de la fecha dentro de `datos.time`.
        mostrar (bool): Si es False, devuelve la figura en vez de mostrarla.
    """
    imagen = datos.isel(time=indice_tiempo)
    if modo == "RGB":
        fig = px.imshow(crear_imagen_rgb(imagen, ["red", "green", "blue"], 3.0))
    elif modo == "Falso color":
        fig = px.imshow(crear_imagen_rgb(imagen, ["nir08", "red", "green"], 4.0))
    else:
        indice = imagen[modo]
        p2, p98 = np.nanpercentile(indice.values, [2, 98])
        fig = px.imshow(indice, color_continuous_scale="RdYlGn", zmin=max(p2, -0.2), zmax=min(p98, 1.0))
    fig.update_layout(title=f"{modo} - {str(datos.time.values[indice_tiempo])[:10]}", width=800, height=600)
    return _finalizar(fig, mostrar)


def explorador_imagenes(datos, modos=("RGB", "Falso color", "NDVI", "EVI")):
    """
    Muestra menús para elegir la fecha y el tipo de imagen (solo en Jupyter/Colab).

    Args:
        datos (xarray.Dataset): Ver `mostrar_imagen_interactiva`.
        modos (iterable): Tipos de imagen disponibles en el menú.
    """
    import ipywidgets as widgets

    fechas = [(str(t)[:10], i) for i, t in enumerate(datos.time.values)]
    widgets.interact(
        lambda indice_tiempo, modo: mostrar_imagen_interactiva(datos, modo, indice_tiempo),
        indice_tiempo=widgets.Dropdown(options=fechas, description="Fecha:"),
        modo=widgets.Dropdown(options=list(modos), description="Imagen:"),
    )


# ---------------------------------------------------------------------------
# Mapas interactivos
# ---------------------------------------------------------------------------
def mapa_bbox(bbox, zoom=10, ancho=800, alto=400):
    """
    Mapa de folium con un rectángulo rojo que marca el área de estudio.

    Incluye mapa base OpenStreetMap, imagen satelital (Google Hybrid) y la
    posición del cursor.

    Args:
        bbox (list): [oeste, sur, este, norte] en grados.
        zoom (int): Zoom inicial.
        ancho, alto (int): Tamaño del mapa en píxeles.

    Returns:
        folium.Map: Escribe el nombre de la variable al final de la celda para verlo.
    """
    import folium
    from folium.plugins import MousePosition

    oeste, sur, este, norte = bbox
    mapa = folium.Map(
        location=[(sur + norte) / 2, (oeste + este) / 2],
        zoom_start=zoom,
        tiles=None,
        width=ancho,
        height=alto,
    )
    MousePosition(position="bottomleft").add_to(mapa)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(mapa)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Hybrid",
        overlay=False,
    ).add_to(mapa)
    folium.Rectangle(
        bounds=[[sur, oeste], [norte, este]], color="red", fill=False, weight=2
    ).add_to(mapa)
    folium.LayerControl().add_to(mapa)
    return mapa


def _imagen_a_base64(imagen_rgb):
    """Convierte un arreglo RGB en una imagen PNG incrustada (data URL)."""
    buffer = io.BytesIO()
    plt.imsave(buffer, imagen_rgb, format="png")
    texto = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{texto}"


class SelectorRegion:
    """
    Mapa interactivo para dibujar un rectángulo sobre una imagen RGB.

    Uso:
        selector = SelectorRegion(rgb, bbox)
        selector.mostrar()          # dibuja el rectángulo con el mouse
        selector.limites            # {"lon_min", "lon_max", "lat_min", "lat_max"} o None

    Atributos:
        limites (dict o None): Rectángulo dibujado, en grados (EPSG:4326).
    """

    def __init__(self, imagen_rgb, bbox, zoom=12, tamano="600px"):
        from ipyleaflet import DrawControl, ImageOverlay, Map
        from ipywidgets import Output

        self.limites = None
        self.salida = Output()

        oeste, sur, este, norte = bbox
        self.mapa = Map(
            center=[(sur + norte) / 2, (oeste + este) / 2],
            zoom=zoom,
            layout=dict(width=tamano, height=tamano),
        )
        self.mapa.add_layer(
            ImageOverlay(
                url=_imagen_a_base64(imagen_rgb),
                bounds=[[sur, oeste], [norte, este]],
                opacity=0.8,
            )
        )
        dibujo = DrawControl(rectangle={"shapeOptions": {"color": "red", "weight": 3}})
        dibujo.on_draw(self._al_dibujar)
        self.mapa.add_control(dibujo)

    def _al_dibujar(self, target, action, geo_json):
        """Guarda el rectángulo cuando el usuario termina de dibujarlo."""
        if action != "created" or geo_json["geometry"]["type"] != "Polygon":
            return
        vertices = geo_json["geometry"]["coordinates"][0]
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        self.limites = {
            "lon_min": min(lons),
            "lon_max": max(lons),
            "lat_min": min(lats),
            "lat_max": max(lats),
        }
        with self.salida:
            self.salida.clear_output(wait=True)
            print("Región seleccionada:")
            print(f"  ({self.limites['lon_min']:.4f}, {self.limites['lat_min']:.4f}) a "
                  f"({self.limites['lon_max']:.4f}, {self.limites['lat_max']:.4f})")

    def mostrar(self):
        """Muestra el mapa y el texto con la región seleccionada."""
        from IPython.display import display

        print("Dibuja un rectángulo en el mapa para seleccionar una región.")
        display(self.mapa)
        display(self.salida)
