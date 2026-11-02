"""Validación de esquemas y calidad de datos para las fuentes de entrada."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Esquemas esperados por fuente
# ---------------------------------------------------------------------------

ESQUEMAS: dict[str, dict[str, str]] = {
    "train": {
        "id": "int64",
        "date": "object",
        "store_nbr": "int64",
        "family": "object",
        "sales": "float64",
        "onpromotion": "int64",
    },
    "stores": {
        "store_nbr": "int64",
        "city": "object",
        "state": "object",
        "type": "object",
        "cluster": "int64",
    },
    "holidays_events": {
        "date": "object",
        "type": "object",
        "locale": "object",
        "locale_name": "object",
        "description": "object",
        "transferred": "object",
    },
    "oil": {
        "date": "object",
        "dcoilwtico": "object",  # puede tener nulos al leerlo como object primero
    },
    "transactions": {
        "date": "object",
        "store_nbr": "int64",
        "transactions": "int64",
    },
}

STORES_VALIDOS = set(range(1, 55))  # 54 tiendas
FAMILIAS_VALIDAS: set[str] | None = None  # se carga la primera vez


@dataclass
class ResultadoValidacion:
    """Resultado de validar un archivo fuente."""
    fuente: str
    archivo: Path
    filas: int
    errores: list[str]

    @property
    def valido(self) -> bool:
        return len(self.errores) == 0


def validar_columnas(df: pd.DataFrame, fuente: str) -> list[str]:
    """Verifica que el DataFrame tenga las columnas esperadas."""
    errores: list[str] = []
    esperadas = set(ESQUEMAS.get(fuente, {}).keys())
    presentes = set(df.columns)
    faltantes = esperadas - presentes
    if faltantes:
        errores.append(f"Columnas faltantes en '{fuente}': {sorted(faltantes)}")
    return errores


def validar_ventas(df: pd.DataFrame) -> list[str]:
    """Validaciones específicas del archivo de ventas (train/bandeja de ventas)."""
    errores: list[str] = []
    if "sales" in df.columns:
        negativos = (df["sales"] < 0).sum()
        if negativos > 0:
            errores.append(f"{negativos} filas con ventas negativas (valores inesperados)")
    if "store_nbr" in df.columns:
        invalidos = set(df["store_nbr"].unique()) - STORES_VALIDOS
        if invalidos:
            errores.append(f"store_nbr inválidos: {sorted(invalidos)}")
    if "date" in df.columns:
        try:
            pd.to_datetime(df["date"])
        except Exception as exc:
            errores.append(f"Fechas no parseables en columna 'date': {exc}")
    return errores


def validar_archivo(ruta: Path, fuente: str) -> ResultadoValidacion:
    """
    Lee y valida un archivo CSV de una fuente dada.

    Args:
        ruta: Ruta al archivo CSV.
        fuente: Nombre de la fuente (clave en ESQUEMAS).

    Returns:
        ResultadoValidacion con los errores encontrados (lista vacía si es válido).
    """
    errores: list[str] = []

    if not ruta.exists():
        return ResultadoValidacion(fuente, ruta, 0, [f"Archivo no encontrado: {ruta}"])

    try:
        df = pd.read_csv(ruta, low_memory=False)
    except Exception as exc:
        return ResultadoValidacion(fuente, ruta, 0, [f"Error leyendo CSV: {exc}"])

    errores.extend(validar_columnas(df, fuente))

    if fuente in ("train", "ventas"):
        errores.extend(validar_ventas(df))

    if df.empty:
        errores.append("El archivo está vacío (0 filas)")

    resultado = ResultadoValidacion(fuente, ruta, len(df), errores)
    if resultado.valido:
        logger.info("✓ %s: %d filas, sin errores", fuente, resultado.filas)
    else:
        logger.warning("✗ %s: %d errores — %s", fuente, len(errores), errores)

    return resultado
