"""
Funciones de apoyo para los notebooks de Geomática Aplicada.

Cada módulo agrupa un tema y se importa explícitamente:

    from utils.vector import descargar_natural_earth, calcular_area
    from utils.stac import conectar_catalogo, parametros_busqueda, parametros_carga, preparar_items
    from utils.raster import preparar_reflectancia, calcular_indices
    from utils.viz import crear_imagen_rgb, mostrar_imagen
    from utils.series import cargar_puntos_csv, cargar_series, tendencia_lineal
    from utils.extremos import cargar_landsat, climatologia_ventana, detectar_episodios
    from utils.ia import herramientas, usar, preguntar

Para que `import utils` funcione, el notebook debe ejecutar antes la celda
"Preparar entorno" (ver utils/README.md).
"""
