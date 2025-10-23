# %% [markdown]
# # Ejercicio: Histogramas Landsat Interactivo
#
# Descarga una imagen Landsat, calcula histogramas de todas las bandas y selecciona una región interactivamente.

# %%
# @title Instalación de paquetes
# %pip install rioxarray xarray matplotlib numpy planetary-computer pystac-client odc-stac ipyleaflet plotly pyproj ipython

# %%
# @title Imports
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
import plotly.graph_objects as go
import xarray as xr
import rioxarray as rxr
import base64
import io
from IPython.display import display
from ipyleaflet import DrawControl, ImageOverlay, Map
from ipywidgets import Output
from pystac_client import Client
from pyproj import Transformer


plt.rcParams["figure.figsize"] = (12, 8)
rxr

# %%
# @title Cargar Landsat
# Área de interés: Valle Central, Chile
bbox = [-70.8, -33.8, -70.6, -33.6]  # [oeste, sur, este, norte]

# Conectar a STAC
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Buscar Landsat 8/9
search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime="2025-01-01/2025-01-31",
    query={
        "eo:cloud_cover": {"lt": 20},
        "platform": {"in": ["landsat-8", "landsat-9"]}
    },
)

items = list(search.get_items())
print(f"Encontradas {len(items)} imágenes Landsat")

# Cargar todas las bandas ópticas
bandas = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]

ds = odc.stac.load(
    items,
    bands=bandas,
    bbox=bbox,
    crs="EPSG:32719",
    resolution=30,
    groupby="solar_day",
)

# Aplicar scale/offset para reflectancia
item = items[0]
scale = item.assets["red"].to_dict()["raster:bands"][0]["scale"]
offset = item.assets["red"].to_dict()["raster:bands"][0]["offset"]

ds_refl = ds * scale + offset
imagen = ds_refl.isel(time=0)

print(f"Dataset cargado: {dict(imagen.sizes)}")
print(f"Bandas: {list(imagen.data_vars)}")

# %%
# @title Visualizar RGB
def crear_rgb(imagen, factor=3.0):
    r = imagen.red.values
    g = imagen.green.values
    b = imagen.blue.values
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * factor, 0, 1)

rgb_img = crear_rgb(imagen)

plt.figure(figsize=(12, 10))
plt.imshow(rgb_img)
plt.title(f"Imagen Landsat - {imagen.time.values}")
plt.axis("off")
plt.show()

# %%
# @title Histograma Completo (Plotly)
def plot_histogramas(dataset, titulo="Histogramas"):
    fig = go.Figure()
    
    # Colores que coinciden con las bandas espectrales
    colores = {
        'coastal': '#87CEEB',    # Light blue (coastal)
        'blue': '#0000FF',       # Blue
        'green': '#00FF00',      # Green
        'red': '#FF0000',        # Red
        'nir08': '#8B0000',      # Dark red (NIR)
        'swir16': '#FFD700',     # Gold (SWIR1)
        'swir22': '#FFA500'      # Orange (SWIR2)
    }
    
    for banda in bandas:
        color = colores[banda]
        datos = dataset[banda].values.flatten()
        datos = datos[~np.isnan(datos)]
        hist, bins = np.histogram(datos, bins=100, range=(0, 0.5))
        
        fig.add_trace(go.Scatter(
            x=bins[:-1], 
            y=hist, 
            mode='lines',
            name=banda,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title='Reflectancia',
        yaxis_title='Frecuencia',
        width=800,
        height=500
    )
    
    fig.show()

plot_histogramas(imagen, "Histogramas - Escena Completa")

# %%
# @title Selección Interactiva (ipyleaflet)

# Convertir imagen RGB a base64
def image_to_base64(img_array):
    buf = io.BytesIO()
    plt.imsave(buf, img_array, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

# Crear mapa con aspecto más cuadrado
center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
m = Map(center=center, zoom=12, layout=dict(width='600px', height='600px'))

# Agregar imagen como overlay usando base64
img_data_url = image_to_base64(rgb_img)
img_overlay = ImageOverlay(
    url=img_data_url,
    bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
    opacity=0.8
)
m.add_layer(img_overlay)

# Control de dibujo
draw = DrawControl(rectangle={'shapeOptions': {'color': 'red', 'weight': 3}})

# Variables globales para almacenar selección
selected_bounds = None
selected_pixels = None

# Widget de output para mostrar solo el texto
output_widget = Output()

def handle_draw(target, action, geo_json):
    global selected_bounds, selected_pixels
    
    if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
        coords = geo_json['geometry']['coordinates'][0]
        
        # Extraer coordenadas del rectángulo
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)
        
        # Guardar coordenadas geográficas para usar en el corte
        selected_bounds = {
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }
        
        # Limpiar solo el output del texto y mostrar el último
        with output_widget:
            output_widget.clear_output(wait=True)
            print(f"Región seleccionada:")
            print(f"  Coordenadas: ({lon_min:.4f}, {lat_min:.4f}) a ({lon_max:.4f}, {lat_max:.4f})")
            print("✅ Región guardada. Ejecuta la siguiente celda para ver histogramas.")

draw.on_draw(handle_draw)
m.add_control(draw)

print("🗺️ Dibuja un rectángulo en el mapa para seleccionar una región")
display(m)
display(output_widget)

# %%
# @title Histograma Región Seleccionada
if selected_bounds is not None:
    
    # Las coordenadas del rectángulo están en WGS84 (EPSG:4326)
    # La imagen está en UTM Zone 19S (EPSG:32719)
    # Necesitamos transformar las coordenadas
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
    
    # Transformar esquinas del rectángulo a UTM
    x_min_utm, y_min_utm = transformer.transform(selected_bounds['lon_min'], selected_bounds['lat_min'])
    x_max_utm, y_max_utm = transformer.transform(selected_bounds['lon_max'], selected_bounds['lat_max'])
    
    print(f"Coordenadas WGS84: ({selected_bounds['lon_min']:.4f}, {selected_bounds['lat_min']:.4f}) a ({selected_bounds['lon_max']:.4f}, {selected_bounds['lat_max']:.4f})")
    print(f"Coordenadas UTM: ({x_min_utm:.0f}, {y_min_utm:.0f}) a ({x_max_utm:.0f}, {y_max_utm:.0f})")
    
    # Recortar imagen usando coordenadas UTM
    region = imagen.sel(
        x=slice(x_min_utm, x_max_utm),
        y=slice(y_max_utm, y_min_utm)  # Invertir y para UTM
    )
    
    print(f"Región recortada: {dict(region.sizes)}")
    
    # Calcular histogramas para la región
    plot_histogramas(region, "Histogramas - Región Seleccionada")
    
    # Mostrar la región recortada
    rgb_region = crear_rgb(region)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_region)
    plt.title("Región Seleccionada - Color Verdadero")
    plt.axis("off")
    plt.show()
    
else:
    print("❌ Primero selecciona una región dibujando un rectángulo en el mapa")

# %%
# @title Export GeoTIFF
# Combinar todas las bandas en un solo DataArray
banda_stack = xr.concat([imagen[b] for b in bandas], dim='band')
banda_stack = banda_stack.assign_coords(band=bandas)

# Exportar como GeoTIFF multi-banda
banda_stack.rio.to_raster('landsat_scene.tif', driver='GTiff')

# Mostrar información del archivo
print(f"\nInformación del archivo:")
print(f"  Dimensiones: {banda_stack.sizes}")
print(f"  Bandas: {list(banda_stack.band.values)}")
print(f"  CRS: {banda_stack.rio.crs}")
print(f"  Resolución: {banda_stack.rio.resolution()}")