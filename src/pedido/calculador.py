"""Cálculo del pedido sugerido a partir del pronóstico y los costos de negocio."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calcular_pedido_sugerido(
    pronostico: pd.DataFrame,
    stockout_cost: float,
    waste_cost: float,
    order_quantile: float = 0.7,
    col_pronostico: str = "sales_pred",
    col_pedido: str = "pedido_sugerido",
) -> pd.DataFrame:
    """
    Calcula la cantidad sugerida a pedir por tienda y familia.

    Lógica:
    - Si el costo de quiebre > costo de merma: usar cuantil alto del pronóstico.
    - El pedido nunca puede ser negativo.
    - Se redondea al entero superior (unidades enteras).

    El cuantil óptimo teórico en un modelo newsvendor es:
        q* = stockout_cost / (stockout_cost + waste_cost)

    Args:
        pronostico: DataFrame con pronóstico por [date, store_nbr, family].
        stockout_cost: Costo por unidad de quiebre de stock.
        waste_cost: Costo por unidad de merma.
        order_quantile: Cuantil a usar para el pedido (sobreescribe el óptimo si se pasa).
        col_pronostico: Columna con el pronóstico central.
        col_pedido: Nombre de la columna de salida.

    Returns:
        DataFrame con la columna de pedido sugerido agregada.
    """
    df = pronostico.copy()

    # Cuantil óptimo de newsvendor (si no se sobreescribe)
    q_optimo = stockout_cost / (stockout_cost + waste_cost + 1e-9)
    logger.info(
        "Cuantil newsvendor óptimo: %.3f | usando: %.3f",
        q_optimo, order_quantile,
    )

    # Factor de ajuste por cuantil: si order_quantile > 0.5, pedimos más que la mediana
    # Asumimos distribución de error aproximadamente normal sobre el pronóstico
    from scipy.stats import norm  # import diferido para no requerir scipy en tests simples
    factor = norm.ppf(order_quantile)  # z-score del cuantil

    # Estimamos std del error como 30% del pronóstico (ajustable con datos reales)
    if f"{col_pronostico}_std" in df.columns:
        std_pred = df[f"{col_pronostico}_std"]
    else:
        std_pred = df[col_pronostico] * 0.30

    df[col_pedido] = np.ceil(
        np.clip(df[col_pronostico] + factor * std_pred, 0, None)
    ).astype(int)

    logger.info(
        "Pedido calculado: mín=%d, máx=%d, promedio=%.1f",
        df[col_pedido].min(), df[col_pedido].max(), df[col_pedido].mean(),
    )
    return df


def agregar_horizonte(
    ultimo_historico: pd.Timestamp,
    horizon_days: int,
) -> pd.DatetimeIndex:
    """
    Genera el índice de fechas futuras para el horizonte de pronóstico.

    Args:
        ultimo_historico: Última fecha en el histórico de ventas.
        horizon_days: Número de días a pronosticar.

    Returns:
        DatetimeIndex con las fechas futuras (excluye el último histórico).
    """
    return pd.date_range(
        start=ultimo_historico + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )


def construir_grilla_prediccion(
    stores: list[int],
    families: list[str],
    fechas: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Construye el DataFrame vacío con todas las combinaciones
    (fecha × tienda × familia) para predecir.

    Args:
        stores: Lista de números de tienda.
        families: Lista de familias de producto.
        fechas: Fechas del horizonte.

    Returns:
        DataFrame con columnas [date, store_nbr, family].
    """
    from itertools import product
    filas = [
        {"date": fecha, "store_nbr": store, "family": family}
        for fecha, store, family in product(fechas, stores, families)
    ]
    return pd.DataFrame(filas)
