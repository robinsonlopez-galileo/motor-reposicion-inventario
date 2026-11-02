"""
DAG de entrenamiento y generación del pronóstico semanal + pedido sugerido.

Ejecuta cada semana (lunes). Entrena el modelo, evalúa contra el vigente,
y publica el pronóstico + pedido sugerido en data/forecasts/.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FORECASTS = PROJECT_ROOT / "data" / "forecasts"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_ARGS = {
    "owner": "robinson.lopez",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}


@dag(
    dag_id="pronostico_semanal",
    description="Entrena el modelo, compara con el vigente y publica pronóstico + pedido",
    schedule="@weekly",    # cada lunes
    start_date=days_ago(7),
    catchup=True,
    default_args=DEFAULT_ARGS,
    tags=["modelo", "pronostico"],
)
def pronostico_semanal():

    @task()
    def entrenar_y_evaluar(fecha_str: str) -> dict:
        """
        Entrena el modelo nuevo y calcula sus métricas.
        Retorna un dict con las métricas y la ruta del modelo serializado.
        """
        import sys
        import json
        import pandas as pd
        import yaml
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from modelo.entrenador import (
            ModeloLGBM, preparar_xy, split_temporal, calcular_metricas
        )

        # Cargar configuración
        with open(CONFIG_DIR / "model_params.yaml") as f:
            cfg = yaml.safe_load(f)
        with open(CONFIG_DIR / "business_costs.yaml") as f:
            costs_cfg = yaml.safe_load(f)
        costs = costs_cfg["costs"]

        # Cargar features
        ruta_features = DATA_PROCESSED / f"features_{fecha_str}.parquet"
        if not ruta_features.exists():
            raise FileNotFoundError(f"Features no encontradas: {ruta_features}")
        df = pd.read_parquet(ruta_features)
        df["date"] = pd.to_datetime(df["date"])

        # Split temporal: últimas 4 semanas = validación
        cutoff = pd.Timestamp(fecha_str)
        fecha_val = cutoff - pd.Timedelta(days=28)
        train_df, val_df = split_temporal(df, fecha_val)

        X_train, y_train = preparar_xy(train_df)
        X_val, y_val = preparar_xy(val_df)

        # Entrenar modelo nuevo
        modelo = ModeloLGBM(cfg["lightgbm"])
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_val)
        metricas = calcular_metricas(y_val, pred, costs)

        # Guardar modelo candidato
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ruta_modelo = MODELS_DIR / f"lgbm_{fecha_str}.pkl"
        modelo.guardar(ruta_modelo)

        return {
            "metricas": metricas,
            "ruta_modelo": str(ruta_modelo),
            "fecha": fecha_str,
        }

    @task()
    def seleccionar_modelo(resultado: dict) -> str:
        """
        Compara el modelo nuevo con el vigente y actualiza el symlink 'best_model.pkl'
        si el nuevo es mejor según el umbral configurado.
        """
        import sys
        import yaml
        import os
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from modelo.entrenador import debe_reemplazar_modelo

        with open(CONFIG_DIR / "model_params.yaml") as f:
            cfg = yaml.safe_load(f)

        metricas_nuevo = resultado["metricas"]
        best_model_path = MODELS_DIR / "best_model.pkl"
        metricas_path = MODELS_DIR / "best_model_metricas.json"

        if metricas_path.exists():
            import json
            with open(metricas_path) as f:
                metricas_vigente = json.load(f)
        else:
            metricas_vigente = {"RMSE": float("inf")}

        if debe_reemplazar_modelo(
            metricas_nuevo,
            metricas_vigente,
            cfg["replacement_threshold_pct"],
        ):
            # Actualizar best_model.pkl como symlink al nuevo
            if best_model_path.exists() or best_model_path.is_symlink():
                best_model_path.unlink()
            best_model_path.symlink_to(Path(resultado["ruta_modelo"]).name)

            import json
            with open(metricas_path, "w") as f:
                json.dump(metricas_nuevo, f, indent=2)
            return resultado["ruta_modelo"]
        else:
            return str(best_model_path.resolve())

    @task()
    def publicar_pronostico(ruta_modelo: str, fecha_str: str) -> str:
        """
        Genera el pronóstico para el horizonte configurado y calcula el pedido sugerido.
        Escribe data/forecasts/pronostico_YYYY-MM-DD.parquet (idempotente).
        """
        import sys
        import yaml
        import pandas as pd
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from modelo.entrenador import ModeloLGBM, preparar_xy
        from pedido.calculador import (
            agregar_horizonte, construir_grilla_prediccion, calcular_pedido_sugerido
        )
        from features.constructor import (
            agregar_features_temporales, agregar_lags, agregar_medias_moviles,
            unir_features_externas, filtrar_hasta_corte
        )

        with open(CONFIG_DIR / "model_params.yaml") as f:
            cfg_model = yaml.safe_load(f)
        with open(CONFIG_DIR / "business_costs.yaml") as f:
            cfg_costs = yaml.safe_load(f)
        costs = cfg_costs["costs"]

        # Cargar modelo
        modelo = ModeloLGBM.cargar(Path(ruta_modelo), cfg_model["lightgbm"])

        # Cargar datos para construir features del horizonte
        ventas = pd.read_parquet(DATA_RAW / "ventas.parquet")
        oil = pd.read_parquet(DATA_RAW / "oil.parquet")
        holidays = pd.read_parquet(DATA_RAW / "holidays_events.parquet")
        stores = pd.read_parquet(DATA_RAW / "stores.parquet")
        for df in [ventas, oil, holidays]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        cutoff = pd.Timestamp(fecha_str)
        horizon = cfg_costs["forecast_horizon_days"]

        stores_list = sorted(ventas["store_nbr"].unique().tolist())
        families_list = sorted(ventas["family"].unique().tolist())
        fechas_futuras = agregar_horizonte(cutoff, horizon)

        # Grilla de predicción
        grilla = construir_grilla_prediccion(stores_list, families_list, fechas_futuras)

        # Combinar histórico + grilla para calcular features (lags necesitan histórico)
        ventas_corte = filtrar_hasta_corte(ventas, cutoff)
        ventas_corte["sales"] = ventas_corte["sales"]  # columna presente
        grilla["sales"] = float("nan")  # placeholder

        combinado = pd.concat([ventas_corte, grilla], ignore_index=True)
        combinado = agregar_features_temporales(combinado)
        combinado = agregar_lags(combinado)
        combinado = agregar_medias_moviles(combinado)

        oil_corte = filtrar_hasta_corte(oil, cutoff)
        holidays_corte = filtrar_hasta_corte(holidays, cutoff)
        combinado = unir_features_externas(combinado, oil_corte, holidays_corte, stores)

        # Filtrar solo el horizonte futuro para predecir
        futuro = combinado[combinado["date"].isin(fechas_futuras)].copy()
        X_futuro, _ = preparar_xy(futuro)
        futuro["sales_pred"] = modelo.predict(X_futuro)
        futuro["cutoff_date"] = cutoff

        # Calcular pedido sugerido
        resultado = calcular_pedido_sugerido(
            futuro,
            stockout_cost=costs["stockout_cost_per_unit"],
            waste_cost=costs["waste_cost_per_unit"],
            order_quantile=cfg_costs["order_quantile"],
        )

        cols_salida = ["cutoff_date", "date", "store_nbr", "family", "sales_pred", "pedido_sugerido"]
        DATA_FORECASTS.mkdir(parents=True, exist_ok=True)
        ruta_salida = DATA_FORECASTS / f"pronostico_{fecha_str}.parquet"
        resultado[cols_salida].to_parquet(ruta_salida, index=False)

        return str(ruta_salida)

    from airflow.operators.python import get_current_context

    @task()
    def obtener_fecha() -> str:
        ctx = get_current_context()
        return ctx["ds"]

    fecha = obtener_fecha()
    resultado_entrenamiento = entrenar_y_evaluar(fecha)
    ruta_modelo = seleccionar_modelo(resultado_entrenamiento)
    publicar_pronostico(ruta_modelo, fecha)


pronostico_semanal()
