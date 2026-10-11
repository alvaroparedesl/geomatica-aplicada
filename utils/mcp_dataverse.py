"""Servidor MCP minimo para Dataverse (se ejecuta en local, dentro de Colab).

Por qué existe: los servidores MCP públicos de Dataverse devuelven el JSON
completo de la API (decenas de KB por búsqueda) y no permiten filtros (`fq`),
facetas (conteos por año, licencia, tema) ni paginación. Este servidor llama a
la misma API pública de búsqueda de Dataverse (no necesita clave para datos
públicos) y devuelve un resumen corto con el TOTAL de resultados, las FACETAS y,
para un dataset, los campos de metadatos que están VACÍOS.

No hace falta ejecutarlo a mano: `utils/ia.py` lo lanza con el alias
"dataverse_local". Código de unas 150 líneas a propósito: sirve también para
enseñar qué es un servidor MCP.
"""

from __future__ import annotations

import json
import re

import requests

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Servidor
except ImportError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Servidor

SERVIDOR = _Servidor("dataverse-local")

HOST_DEFECTO = "dataverse.harvard.edu"
_HOST_OK = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$")
_FACETAS = {  # nombre en la API -> etiqueta mostrada
    "publicationDate": "Año de publicación",
    "license": "Licencia",
    "subject_ss": "Materia",
    "dvCategory": "Tipo de colección",
    "fileTypeGroupFacet": "Tipo de archivo",
}


def _get(host: str, ruta: str, params: dict) -> dict:
    host = host.lower().removeprefix("https://").strip("/")
    if not _HOST_OK.match(host):
        raise ValueError(f"host no válido: {host!r}")
    r = requests.get(f"https://{host}{ruta}", params=params, timeout=60,
                     headers={"User-Agent": "curso-geomatica-colab"})
    if r.status_code != 200:  # texto claro para el modelo (puede corregir la consulta y reintentar)
        raise ValueError(f"{host} respondió HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["data"]


def _seguro(f):
    """Convierte excepciones en texto 'ERROR: ...' para que el modelo las lea."""
    import functools

    @functools.wraps(f)
    def envoltorio(*a, **k):
        try:
            return f(*a, **k)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
    return envoltorio


@SERVIDOR.tool()
@_seguro
def dataverse_search(q: str = "*", host: str = HOST_DEFECTO, type: str = "dataset",
                     fq: list[str] = [], sort: str = "score", order: str = "desc",
                     per_page: int = 10, start: int = 0) -> str:
    """Busca en una instalación de Dataverse (Harvard, datav.udec.cl, datos.uchile.cl,
    datos.usach.cl, datospararesiliencia.cl...). Devuelve el TOTAL de resultados, una
    lista corta y FACETAS sobre todo el conjunto (por año, licencia, materia).

    q: consulta Solr. Varios términos se combinan con AND si pones AND; usa comillas
       para frases. Campos útiles: title, authorName, keywordValue, subject,
       country, state, city, otherGeographicCoverage, publicationDate (año), dateSort.
    fq: filtros que no afectan al ranking, p. ej. ["publicationDate:2026", "country:Chile"].
    type: dataset, file o dataverse. sort: score, name, date. per_page: máximo 25.
    """
    params: dict = {"q": q, "type": type, "per_page": max(1, min(per_page, 25)), "start": start,
                    "sort": sort, "order": order, "show_facets": "true"}
    if fq:
        params["fq"] = fq
    d = _get(host, "/api/search", params)
    total, items = d["total_count"], d.get("items", [])
    out = [f"Total: {total} resultados en {host} para q={q!r} fq={fq or []} "
           f"(mostrando {start + 1 if items else 0}-{start + len(items)})"]
    for it in items:
        desc = " ".join((it.get("description") or "").split())[:200]
        out.append(
            f"- {it.get('global_id') or it.get('identifier')} | {(it.get('published_at') or '')[:10]} | "
            f"{it.get('name')} | colección: {it.get('name_of_dataverse', '-')} | "
            f"palabras clave: {', '.join(it.get('keywords') or []) or 'no consta'} | "
            f"materias: {', '.join(it.get('subjects') or []) or 'no consta'} | {desc}")
    for bloque in d.get("facets", []):
        for clave, v in bloque.items():
            if clave in _FACETAS:
                etiquetas = ", ".join(f"{k} ({n})" for lab in v["labels"][:8] for k, n in lab.items())
                out.append(f"Facetas, {_FACETAS[clave]} (sobre el total de {total}): {etiquetas}")
    if total > start + len(items):
        out.append(f"Parcial: faltan {total - start - len(items)} resultados (usa start={start + len(items)}).")
    return "\n".join(out)


def _campos(version: dict) -> dict:
    """Aplana los bloques de metadatos de un dataset en {campo: valor}."""
    res: dict = {}
    for bloque in version.get("metadataBlocks", {}).values():
        for f in bloque.get("fields", []):
            res[f["typeName"]] = f["value"]
    return res


def _valores(lista, clave):
    return [x[clave]["value"] for x in lista or [] if clave in x]


@SERVIDOR.tool()
@_seguro
def dataverse_dataset(persistent_id: str, host: str = HOST_DEFECTO) -> str:
    """Metadatos de UN dataset (persistent_id tipo 'doi:10.7910/DVN/XXXXXX'): título,
    licencia, palabras clave, cobertura geográfica, número de archivos y una lista de
    CAMPOS VACÍOS (lo que un buscador, o una IA, no podrá usar)."""
    d = _get(host, "/api/datasets/:persistentId", {"persistentId": persistent_id})
    v = d["latestVersion"]
    c = _campos(v)
    lic = v.get("license")
    lic = (lic.get("name") if isinstance(lic, dict) else lic) or v.get("termsOfUse") or None
    claves = _valores(c.get("keyword"), "keywordValue")
    geo = c.get("geographicCoverage") or []
    pais = _valores(geo, "country")
    estado = _valores(geo, "state")
    ciudad = _valores(geo, "city")
    otra = _valores(geo, "otherGeographicCoverage")
    caja = bool(c.get("geographicBoundingBox"))
    desc = " ".join(" ".join(_valores(c.get("dsDescription"), "dsDescriptionValue")).split())
    archivos = v.get("files", [])
    vacios = []
    if not claves: vacios.append("palabras clave")
    if not (pais or estado or ciudad or otra): vacios.append("cobertura geográfica (país/región/ciudad)")
    if not caja: vacios.append("caja geográfica (coordenadas)")
    if not lic: vacios.append("licencia")
    if len(desc) < 150: vacios.append(f"descripción corta ({len(desc)} caracteres)")
    if not c.get("timePeriodCovered"): vacios.append("período temporal cubierto")
    return "\n".join([
        f"Título: {c.get('title')}",
        f"ID: {persistent_id} | publicado: {(v.get('releaseTime') or '')[:10]} | versión {v.get('versionNumber')}.{v.get('versionMinorNumber')}",
        f"Licencia: {lic or 'no consta'}",
        f"Palabras clave: {', '.join(claves) or 'no consta'}",
        f"Materias: {', '.join(c.get('subject') or []) or 'no consta'}",
        f"Cobertura geográfica: país={pais or 'no consta'}, región={estado or 'no consta'}, "
        f"ciudad={ciudad or 'no consta'}, otra={otra or 'no consta'}; caja geográfica: {'sí' if caja else 'no'}",
        f"Archivos: {len(archivos)} ({', '.join(sorted({f['dataFile'].get('contentType', '?') for f in archivos}))})",
        f"Descripción: {desc[:400] or 'no consta'}",
        "CAMPOS VACÍOS: " + (", ".join(vacios) if vacios else "ninguno de los comprobados"),
    ])


if __name__ == "__main__":
    SERVIDOR.run()  # transporte stdio
