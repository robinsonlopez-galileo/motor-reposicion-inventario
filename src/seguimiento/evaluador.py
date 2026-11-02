"""Seguimiento del desempeño: compara pronósticos publicados con ventas reales."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cargar_pronostico(ruta: Path) -> pd.DataFrame:
    """
    Lee un archivo de pronóstico publicado (parquet).

    Columnas esperadas: [cutoff_date, date, store_nbr, family, sales_pred, pedido_sugerido]
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Pronóstico no encontrado: {ruta}")
    df = pd.read_parquet(ruta)
    df["date"] = pd.to_datetime(df["date"])
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df


def comparar_con_reales(
    pronostico: pd.DataFrame,
    ventas_reales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une el pronóstico con las ventas reales disponibles.

    Solo se comparan filas donde las ventas reales ya fueron observadas.

    Args:
        pronostico: DataFrame de pronóstico.
        ventas_reales: DataFrame histórico de ventas (train actualizado).

    Returns:
        DataFrame con columnas [date, store_nbr, family, sales_pred, sales_real, error_abs].
    """
    real = ventas_reales[["date", "store_nbr", "family", "sales"]].copy()
    real = real.rename(columns={"sales": "sales_real"})

    merged = pronostico.merge(
        real,
        on=["date", "store_nbr", "family"],
        how="inner",  # solo fechas con datos reales observados
    )

    merged["error_abs"] = (merged["sales_pred"] - merged["sales_real"]).abs()
    merged["error_relativo_pct"] = (
        merged["error_abs"] / (merged["sales_real"].abs() + 1e-6) * 100
    )
    logger.info(
        "Comparación: %d pares pronóstico-real | MAE=%.3f",
        len(merged), merged["error_abs"].mean(),
    )
    return merged


def resumen_desempeno(comparacion: pd.DataFrame) -> dict[str, float]:
    """
    Genera un resumen de métricas de desempeño a partir de la comparación.

    Returns:
        Diccionario con MAE, RMSE, MAPE_pct.
    """
    import numpy as np

    if comparacion.empty:
        logger.warning("Sin datos para calcular métricas de desempeño")
        return {}

    mae = comparacion["error_abs"].mean()
    rmse = float(np.sqrt((comparacion["error_abs"] ** 2).mean()))
    mape = comparacion["error_relativo_pct"].mean()

    resumen = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_pct": round(mape, 2)}
    logger.info("Desempeño: %s", resumen)
    return resumen


def guardar_reporte(
    comparacion: pd.DataFrame,
    resumen: dict[str, float],
    ruta_reporte: Path,
    cutoff_date: pd.Timestamp,
) -> None:
    """
    Guarda el reporte de seguimiento en parquet y el resumen en CSV.

    Args:
        comparacion: DataFrame detallado de comparación.
        resumen: Diccionario de métricas.
        ruta_reporte: Directorio donde guardar los archivos.
        cutoff_date: Fecha de corte del pronóstico evaluado.
    """
    ruta_reporte.mkdir(parents=True, exist_ok=True)
    fecha_str = cutoff_date.strftime("%Y-%m-%d")

    detalle_path = ruta_reporte / f"detalle_{fecha_str}.parquet"
    comparacion.to_parquet(detalle_path, index=False)

    resumen_df = pd.DataFrame([{**resumen, "cutoff_date": fecha_str}])
    resumen_path = ruta_reporte / "historial_desempeno.csv"
    if resumen_path.exists():
        historico = pd.read_csv(resumen_path)
        resumen_df = pd.concat([historico, resumen_df], ignore_index=True)
    resumen_df.to_csv(resumen_path, index=False)

    logger.info("Reporte guardado en %s", ruta_reporte)
