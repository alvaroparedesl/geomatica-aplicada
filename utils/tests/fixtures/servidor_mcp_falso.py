"""Servidor MCP mínimo (stdio) para probar `utils/ia.py` sin internet.

Ofrece dos herramientas con respuestas fijas y una tercera que falla, para
comprobar que el cliente muestra los pasos y que un error llega al modelo como
texto. Lo lanzan las pruebas de `test_ia.py` como subproceso.
"""

from __future__ import annotations

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Servidor
except ImportError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Servidor

SERVIDOR = _Servidor("falso")


@SERVIDOR.tool()
def falso_buscar(query: str, size: int = 10) -> str:
    """Busca registros de prueba. Devuelve el total y una lista corta."""
    return f"Total: 3 resultados para {query!r} (size={size})\n- A | 2026\n- B | 2026\n- C | 2025"


@SERVIDOR.tool()
def falso_licencia(doi: str) -> str:
    """Licencia de un registro de prueba."""
    return f"Licencia de {doi}: CC-BY-4.0"


@SERVIDOR.tool()
def falso_error() -> str:
    """Siempre falla."""
    raise ValueError("fallo de prueba")


if __name__ == "__main__":
    SERVIDOR.run()
