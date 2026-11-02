"""Tests para el módulo de modelado."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modelo.entrenador import (
    calcular_metricas,
    debe_reemplazar_modelo,
    split_temporal,
    preparar_xy,
)


def _df_simple() -> pd.DataFrame:
    fechas = pd.date_range("2016-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": fechas,
        "store_nbr": 1,
        "family": "BEVERAGES",
        "sales": np.random.uniform(10, 50, 100),
        "dia_semana": fechas.dayofweek,
        "mes": fechas.month,
        "anio": fechas.year,
        "lag_7": np.random.uniform(10, 50, 100),
        "lag_14": np.random.uniform(10, 50, 100),
        "lag_28": np.random.uniform(10, 50, 100),
        "rolling_mean_7": np.random.uniform(10, 50, 100),
        "rolling_mean_28": np.random.uniform(10, 50, 100),
        "rolling_mean_84": np.random.uniform(10, 50, 100),
    })


def test_split_temporal_sin_solapamiento():
    df = _df_simple()
    fecha_val = pd.Timestamp("2016-03-01")
    train, val = split_temporal(df, fecha_val)
    assert train["date"].max() < fecha_val
    assert val["date"].min() >= fecha_val
    assert len(train) + len(val) == len(df)


def test_split_temporal_train_primero():
    df = _df_simple()
    fecha_val = pd.Timestamp("2016-03-15")
    train, val = split_temporal(df, fecha_val)
    assert train["date"].max() < val["date"].min()


def test_calcular_metricas_perfectas():
    y = pd.Series([10.0, 20.0, 30.0])
    pred = np.array([10.0, 20.0, 30.0])
    metricas = calcular_metricas(y, pred)
    assert metricas["MAE"] == pytest.approx(0.0)
    assert metricas["RMSE"] == pytest.approx(0.0)


def test_calcular_metricas_con_costos():
    y = pd.Series([100.0, 100.0])
    pred = np.array([110.0, 90.0])  # sobrEstima 10, subestima 10
    costs = {"stockout_cost_per_unit": 0.3, "waste_cost_per_unit": 0.1}
    metricas = calcular_metricas(y, pred, costs)
    assert "costo_total" in metricas
    assert metricas["costo_total"] > 0


def test_reemplazar_modelo_si_mejora():
    nuevo = {"RMSE": 8.0}
    vigente = {"RMSE": 10.0}   # mejora de 20%
    assert debe_reemplazar_modelo(nuevo, vigente, threshold_pct=2.0) is True


def test_no_reemplazar_modelo_sin_mejora():
    nuevo = {"RMSE": 9.9}
    vigente = {"RMSE": 10.0}   # mejora de solo 1%
    assert debe_reemplazar_modelo(nuevo, vigente, threshold_pct=2.0) is False


def test_preparar_xy_excluye_target():
    df = _df_simple()
    X, y = preparar_xy(df, col_target="sales")
    assert "sales" not in X.columns
    assert len(y) == len(df)
    assert "date" not in X.columns
