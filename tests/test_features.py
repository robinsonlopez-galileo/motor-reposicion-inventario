"""Tests para el constructor de features."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features.constructor import (
    filtrar_hasta_corte,
    agregar_features_temporales,
    agregar_lags,
    agregar_medias_moviles,
)


def _df_ventas_base() -> pd.DataFrame:
    """DataFrame mínimo de ventas para tests."""
    fechas = pd.date_range("2016-01-01", periods=60, freq="D")
    registros = []
    for fecha in fechas:
        for store in [1, 2]:
            registros.append({
                "date": fecha,
                "store_nbr": store,
                "family": "BEVERAGES",
                "sales": float(np.random.randint(10, 100)),
                "onpromotion": 0,
            })
    return pd.DataFrame(registros)


def test_filtrar_hasta_corte():
    df = _df_ventas_base()
    cutoff = pd.Timestamp("2016-02-01")
    filtrado = filtrar_hasta_corte(df, cutoff)
    assert filtrado["date"].max() <= cutoff
    assert len(filtrado) < len(df)


def test_no_data_leakage():
    """La fecha máxima después del filtro nunca supera el cutoff."""
    df = _df_ventas_base()
    cutoff = pd.Timestamp("2016-02-15")
    filtrado = filtrar_hasta_corte(df, cutoff)
    assert filtrado["date"].max() <= cutoff, "Data leakage detectado"


def test_features_temporales_columnas():
    df = _df_ventas_base()
    resultado = agregar_features_temporales(df)
    for col in ["dia_semana", "mes", "anio", "es_fin_semana", "dia_semana_sin", "dia_semana_cos"]:
        assert col in resultado.columns, f"Falta columna: {col}"


def test_lags_no_leakage():
    """Los lags deben ser NaN al inicio y nunca usar valores del futuro."""
    df = _df_ventas_base()
    resultado = agregar_lags(df, lags=[7], group_cols=["store_nbr", "family"])
    # Para los primeros 7 días de cada grupo, lag_7 debe ser NaN
    primeros = resultado[resultado["date"] < resultado["date"].min() + pd.Timedelta(days=7)]
    assert primeros["lag_7"].isna().all(), "Los primeros 7 días no deberían tener lag_7"


def test_medias_moviles_no_negativas():
    df = _df_ventas_base()
    resultado = agregar_medias_moviles(df, ventanas=[7])
    no_nan = resultado["rolling_mean_7"].dropna()
    assert (no_nan >= 0).all(), "Las medias móviles no pueden ser negativas"
