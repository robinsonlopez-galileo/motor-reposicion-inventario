"""Tests para el calculador de pedido sugerido."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pedido.calculador import (
    agregar_horizonte,
    construir_grilla_prediccion,
    calcular_pedido_sugerido,
)


def test_horizonte_correcto():
    ultimo = pd.Timestamp("2017-01-10")
    fechas = agregar_horizonte(ultimo, 16)
    assert len(fechas) == 16
    assert fechas[0] == pd.Timestamp("2017-01-11")
    assert fechas[-1] == pd.Timestamp("2017-01-26")


def test_grilla_combinaciones():
    stores = [1, 2]
    families = ["BEVERAGES", "PRODUCE"]
    fechas = pd.date_range("2017-01-11", periods=3, freq="D")
    grilla = construir_grilla_prediccion(stores, families, fechas)
    assert len(grilla) == 2 * 2 * 3  # 12 combinaciones
    assert set(grilla.columns) == {"date", "store_nbr", "family"}


def test_pedido_nunca_negativo():
    df = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=5),
        "store_nbr": 1,
        "family": "BEVERAGES",
        "sales_pred": [0.0, -1.0, 5.0, 100.0, 0.5],
    })
    resultado = calcular_pedido_sugerido(df, stockout_cost=0.3, waste_cost=0.1)
    assert (resultado["pedido_sugerido"] >= 0).all()


def test_pedido_entero():
    df = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=3),
        "store_nbr": 1,
        "family": "BEVERAGES",
        "sales_pred": [10.5, 20.3, 5.7],
    })
    resultado = calcular_pedido_sugerido(df, stockout_cost=0.3, waste_cost=0.1)
    assert resultado["pedido_sugerido"].dtype in (int, "int64", np.int64)
