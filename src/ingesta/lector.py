"""Lectura de fuentes desde la bandeja (inbox) y la carga histórica."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columnas de fecha a parsear por fuente
DATE_COLS: dict[str, list[str]] = {
    "train": ["date"],
    "ventas": ["date"],
    "holidays_events": ["date"],
    "oil": ["date"],
    "transactions": ["date"],
}


def leer_fuente(ruta: Path, fuente: str) -> pd.DataFrame:
    """
    Lee un CSV de fuente y convierte las columnas de fecha.

    Principio fundamental: esta función solo lee; no filtra por fecha.
    El corte temporal es responsabilidad del módulo de features.

    Args:
        ruta: Ruta al archivo CSV.
        fuente: Nombre de la fuente para saber qué columnas parsear.

    Returns:
        DataFrame con columnas de fecha como datetime64.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Fuente no encontrada: {ruta}")

    df = pd.read_csv(ruta, low_memory=False)

    if df.empty:
        raise ValueError(f"El archivo está vacío: {ruta}")

    for col in DATE_COLS.get(fuente, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    logger.debug("Leída fuente '%s': %d filas, %d columnas", fuente, len(df), len(df.columns))
    return df


def leer_bandeja(inbox_dir: Path, fuente: str, fecha: str) -> pd.DataFrame | None:
    """
    Lee el archivo de una fuente para una fecha específica desde la bandeja.

    Convención de nombres en la bandeja:
      inbox/<fuente>/<fuente>_YYYY-MM-DD.csv

    Args:
        inbox_dir: Directorio raíz de la bandeja.
        fuente: Nombre de la fuente.
        fecha: Fecha en formato YYYY-MM-DD.

    Returns:
        DataFrame si el archivo existe, None si no hay archivo para esa fecha.
    """
    ruta = inbox_dir / fuente / f"{fuente}_{fecha}.csv"
    if not ruta.exists():
        logger.debug("Sin archivo en bandeja para %s @ %s", fuente, fecha)
        return None
    return leer_fuente(ruta, fuente)


def leer_historico(data_raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Lee la carga histórica completa (archivos consolidados en data/raw/).

    Returns:
        Diccionario {fuente: DataFrame} para las fuentes principales.
    """
    fuentes = ["train", "stores", "holidays_events", "oil", "transactions"]
    resultado: dict[str, pd.DataFrame] = {}
    for fuente in fuentes:
        ruta = data_raw_dir / f"{fuente}.parquet"
        if ruta.exists():
            df = pd.read_parquet(ruta)
            resultado[fuente] = df
            logger.info("Histórico '%s': %d filas", fuente, len(df))
        else:
            logger.warning("No se encontró histórico para '%s' en %s", fuente, data_raw_dir)
    return resultado


def consolidar_nuevos_registros(
    historico: pd.DataFrame,
    nuevos: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Agrega registros nuevos al histórico y elimina duplicados, prefiriendo los nuevos.

    Maneja correcciones: si un registro con las mismas claves ya existe,
    el valor nuevo lo reemplaza (modelo 'last write wins').

    Args:
        historico: DataFrame histórico existente.
        nuevos: DataFrame con registros a incorporar.
        key_cols: Columnas que identifican un registro de forma única.

    Returns:
        DataFrame consolidado sin duplicados.
    """
    if historico.empty:
        return nuevos.copy()

    combinado = pd.concat([historico, nuevos], ignore_index=True)
    # Los nuevos van al final; drop_duplicates con keep='last' los prefiere
    combinado = combinado.drop_duplicates(subset=key_cols, keep="last")
    combinado = combinado.sort_values(key_cols).reset_index(drop=True)
    logger.debug(
        "Consolidado: %d histórico + %d nuevos → %d total",
        len(historico), len(nuevos), len(combinado),
    )
    return combinado
