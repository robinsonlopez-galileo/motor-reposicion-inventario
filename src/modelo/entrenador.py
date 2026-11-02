"""Entrenamiento y evaluación de modelos de pronóstico."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Features que NO son el target ni identificadores
COLS_EXCLUIR = ["date", "id", "sales", "type", "city", "family"]

# Features categóricas que LightGBM puede manejar nativamente
COLS_CATEGORICAS = ["store_nbr", "cluster", "store_type_ord", "dia_semana", "mes"]


def preparar_xy(
    df: pd.DataFrame,
    col_target: str = "sales",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y target (y) del dataset."""
    cols_features = [c for c in df.columns if c not in COLS_EXCLUIR]
    X = df[cols_features].copy()
    y = df[col_target].copy()
    return X, y


def split_temporal(
    df: pd.DataFrame,
    fecha_corte_val: pd.Timestamp,
    col_fecha: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en entrenamiento y validación respetando el orden temporal.

    No hay shuffle: el set de validación siempre es posterior al de entrenamiento.

    Args:
        df: Dataset completo.
        fecha_corte_val: A partir de esta fecha (inclusive) es validación.
        col_fecha: Columna de fecha.

    Returns:
        (train_df, val_df)
    """
    train_df = df[df[col_fecha] < fecha_corte_val].copy()
    val_df = df[df[col_fecha] >= fecha_corte_val].copy()
    logger.info(
        "Split temporal: train=%d filas hasta %s | val=%d filas desde %s",
        len(train_df), fecha_corte_val.date() - pd.Timedelta(days=1),
        len(val_df), fecha_corte_val.date(),
    )
    return train_df, val_df


class ModeloBaseline:
    """
    Baseline: media móvil de las últimas N semanas para el mismo día de la semana.

    No tiene parámetros que aprender; usa el histórico directamente.
    """

    def __init__(self, rolling_weeks: int = 8) -> None:
        self.rolling_weeks = rolling_weeks
        self._historico: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "ModeloBaseline":
        """Almacena el histórico para calcular la media móvil al predecir."""
        self._historico = df[["date", "store_nbr", "family", "sales"]].copy()
        return self

    def predict(self, df_futuro: pd.DataFrame) -> np.ndarray:
        """
        Predice las ventas calculando la media de las últimas N semanas
        para el mismo día de la semana, tienda y familia.
        """
        if self._historico is None:
            raise RuntimeError("Modelo no entrenado; llama a fit() primero.")

        predicciones = []
        for _, fila in df_futuro.iterrows():
            mismo_dia = self._historico[
                (self._historico["store_nbr"] == fila["store_nbr"])
                & (self._historico["family"] == fila["family"])
                & (self._historico["date"].dt.dayofweek == fila["date"].dayofweek())
                & (self._historico["date"] < fila["date"])
            ]
            if len(mismo_dia) == 0:
                predicciones.append(0.0)
            else:
                ultimo = mismo_dia.sort_values("date").tail(self.rolling_weeks)
                predicciones.append(float(ultimo["sales"].mean()))
        return np.array(predicciones)


class ModeloLGBM:
    """Wrapper de LightGBM para pronóstico de ventas."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self._model: Any = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModeloLGBM":
        """Entrena el modelo LightGBM."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm no está instalado: pip install lightgbm") from exc

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X_train, y_train,
            categorical_feature=[c for c in COLS_CATEGORICAS if c in X_train.columns],
        )
        logger.info("LightGBM entrenado con %d filas y %d features", len(X_train), X_train.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Modelo no entrenado.")
        pred = self._model.predict(X)
        return np.clip(pred, 0, None)  # ventas nunca negativas

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Modelo guardado en %s", ruta)

    @classmethod
    def cargar(cls, ruta: Path, params: dict[str, Any]) -> "ModeloLGBM":
        with open(ruta, "rb") as f:
            model = pickle.load(f)
        obj = cls(params)
        obj._model = model
        return obj


def calcular_metricas(
    y_real: pd.Series,
    y_pred: np.ndarray,
    costs: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Calcula métricas técnicas y de negocio.

    Args:
        y_real: Ventas reales.
        y_pred: Pronóstico.
        costs: Diccionario con 'stockout_cost_per_unit' y 'waste_cost_per_unit'.

    Returns:
        Diccionario con métricas.
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mape = float(np.mean(np.abs((y_real - y_pred) / (y_real + 1e-6))) * 100)

    metricas: dict[str, float] = {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}

    if costs:
        error = y_pred - y_real
        costo_merma = float(np.sum(np.clip(error, 0, None)) * costs["waste_cost_per_unit"])
        costo_quiebre = float(np.sum(np.clip(-error, 0, None)) * costs["stockout_cost_per_unit"])
        metricas["costo_merma"] = costo_merma
        metricas["costo_quiebre"] = costo_quiebre
        metricas["costo_total"] = costo_merma + costo_quiebre

    return metricas


def debe_reemplazar_modelo(
    metricas_nuevo: dict[str, float],
    metricas_vigente: dict[str, float],
    threshold_pct: float = 2.0,
) -> bool:
    """
    Decide si el modelo nuevo reemplaza al vigente.

    Criterio: el nuevo reemplaza si su RMSE es al menos threshold_pct% menor.

    Args:
        metricas_nuevo: Métricas del candidato.
        metricas_vigente: Métricas del modelo en producción.
        threshold_pct: Mejora mínima requerida en RMSE (%).

    Returns:
        True si se debe reemplazar el modelo vigente.
    """
    rmse_nuevo = metricas_nuevo.get("RMSE", float("inf"))
    rmse_vigente = metricas_vigente.get("RMSE", float("inf"))
    mejora = (rmse_vigente - rmse_nuevo) / (rmse_vigente + 1e-9) * 100
    reemplazar = mejora >= threshold_pct
    logger.info(
        "RMSE nuevo=%.4f vs vigente=%.4f | mejora=%.2f%% | reemplazar=%s",
        rmse_nuevo, rmse_vigente, mejora, reemplazar,
    )
    return reemplazar
