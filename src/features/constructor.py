"""Construcción de features para el modelo de pronóstico de ventas."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filtrar_hasta_corte(df: pd.DataFrame, cutoff_date: pd.Timestamp, col_fecha: str = "date") -> pd.DataFrame:
    """
    Filtra el DataFrame para incluir solo registros hasta la fecha de corte (inclusive).

    Principio fundamental de no data leakage: ningún feature puede incluir
    información posterior al cutoff_date.

    Args:
        df: DataFrame a filtrar.
        cutoff_date: Fecha máxima de datos permitida.
        col_fecha: Nombre de la columna de fecha.

    Returns:
        DataFrame filtrado.
    """
    return df[df[col_fecha] <= cutoff_date].copy()


def agregar_features_temporales(df: pd.DataFrame, col_fecha: str = "date") -> pd.DataFrame:
    """Agrega variables temporales derivadas de la fecha."""
    df = df.copy()
    df["dia_semana"] = df[col_fecha].dt.dayofweek          # 0=lunes, 6=domingo
    df["dia_mes"] = df[col_fecha].dt.day
    df["semana_anio"] = df[col_fecha].dt.isocalendar().week.astype(int)
    df["mes"] = df[col_fecha].dt.month
    df["anio"] = df[col_fecha].dt.year
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    # Codificación cíclica del día de la semana
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    return df


def agregar_lags(
    df: pd.DataFrame,
    col_target: str = "sales",
    lags: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Agrega columnas de lag (valores pasados) para el target.

    Args:
        df: DataFrame ordenado por fecha dentro de cada grupo.
        col_target: Columna objetivo.
        lags: Días de lag a calcular. Default: [7, 14, 28].
        group_cols: Columnas de agrupación. Default: ['store_nbr', 'family'].

    Returns:
        DataFrame con columnas lag_<n> agregadas.
    """
    if lags is None:
        lags = [7, 14, 28]
    if group_cols is None:
        group_cols = ["store_nbr", "family"]

    df = df.copy().sort_values(group_cols + ["date"])
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(group_cols)[col_target]
            .shift(lag)
        )
    return df


def agregar_medias_moviles(
    df: pd.DataFrame,
    col_target: str = "sales",
    ventanas: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Agrega medias móviles con shift=1 para evitar data leakage.

    Args:
        df: DataFrame.
        col_target: Columna objetivo.
        ventanas: Tamaños de ventana en días. Default: [7, 28, 84].
        group_cols: Columnas de agrupación.

    Returns:
        DataFrame con columnas rolling_mean_<n> agregadas.
    """
    if ventanas is None:
        ventanas = [7, 28, 84]
    if group_cols is None:
        group_cols = ["store_nbr", "family"]

    df = df.copy().sort_values(group_cols + ["date"])
    for v in ventanas:
        df[f"rolling_mean_{v}"] = (
            df.groupby(group_cols)[col_target]
            .transform(lambda x: x.shift(1).rolling(v, min_periods=1).mean())
        )
    return df


def unir_features_externas(
    ventas: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une el DataFrame de ventas con las fuentes externas (petróleo, feriados, tiendas).

    Args:
        ventas: DataFrame de ventas con columnas [date, store_nbr, family, sales, onpromotion].
        oil: DataFrame de precios de petróleo con columnas [date, dcoilwtico].
        holidays: DataFrame de feriados.
        stores: DataFrame de atributos de tiendas.

    Returns:
        DataFrame enriquecido.
    """
    df = ventas.copy()

    # --- Petróleo: interpolar valores faltantes y unir ---
    oil = oil.copy()
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil = oil.set_index("date").reindex(df["date"].unique())
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="time")
    oil = oil.reset_index().rename(columns={"index": "date"})
    df = df.merge(oil[["date", "dcoilwtico"]], on="date", how="left")

    # --- Feriados: flag de feriado nacional ---
    nacionales = (
        holidays[holidays["locale"] == "National"]
        .assign(es_feriado_nacional=1)
        [["date", "es_feriado_nacional"]]
        .drop_duplicates("date")
    )
    df = df.merge(nacionales, on="date", how="left")
    df["es_feriado_nacional"] = df["es_feriado_nacional"].fillna(0).astype(int)

    # --- Tiendas: tipo, ciudad, cluster ---
    df = df.merge(
        stores[["store_nbr", "type", "city", "cluster"]],
        on="store_nbr",
        how="left",
    )
    # Codificar tipo de tienda como ordinal (A=5 mayor volumen, E=1 menor)
    tipo_map = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    df["store_type_ord"] = df["type"].map(tipo_map).fillna(0).astype(int)

    return df


def construir_dataset(
    ventas: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Pipeline completo de construcción del dataset de features.

    Args:
        ventas: DataFrame de ventas.
        oil: DataFrame de petróleo.
        holidays: DataFrame de feriados.
        stores: DataFrame de tiendas.
        cutoff_date: Fecha de corte; no se usa información posterior.

    Returns:
        DataFrame listo para entrenamiento o predicción.
    """
    logger.info("Construyendo dataset con cutoff: %s", cutoff_date.date())

    # 1. Filtrar hasta el corte
    ventas = filtrar_hasta_corte(ventas, cutoff_date)

    # 2. Features temporales
    ventas = agregar_features_temporales(ventas)

    # 3. Lags y medias móviles (calculados dentro del corte → no hay leakage)
    ventas = agregar_lags(ventas)
    ventas = agregar_medias_moviles(ventas)

    # 4. Unir fuentes externas (también filtradas al corte)
    oil_corte = filtrar_hasta_corte(oil, cutoff_date)
    holidays_corte = filtrar_hasta_corte(holidays, cutoff_date)
    ventas = unir_features_externas(ventas, oil_corte, holidays_corte, stores)

    # 5. Eliminar filas con NaN en lags (primeras observaciones sin historial)
    lag_cols = [c for c in ventas.columns if c.startswith("lag_")]
    n_antes = len(ventas)
    ventas = ventas.dropna(subset=lag_cols)
    logger.info("Filas eliminadas por NaN en lags: %d", n_antes - len(ventas))

    logger.info("Dataset construido: %d filas, %d columnas", len(ventas), len(ventas.columns))
    return ventas
