"""
DAG de preparación de datos para el modelo.

Lee data/raw/, construye features y escribe data/processed/.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DEFAULT_ARGS = {
    "owner": "robinson.lopez",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    dag_id="preparacion_datos",
    description="Construye features a partir de data/raw/",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=True,
    default_args=DEFAULT_ARGS,
    tags=["features", "datos"],
)
def preparacion_datos():

    @task()
    def construir_features(fecha_str: str) -> str:
        """
        Lee los datos raw y construye el dataset de features con cutoff = fecha_str.
        Garantiza que no haya data leakage: solo usa info disponible hasta fecha_str.
        """
        import sys
        import pandas as pd
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from features.constructor import construir_dataset

        # Cargar fuentes raw
        def leer(nombre: str) -> pd.DataFrame:
            ruta = DATA_RAW / f"{nombre}.parquet"
            if not ruta.exists():
                raise FileNotFoundError(f"Fuente raw no encontrada: {ruta}")
            df = pd.read_parquet(ruta)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df

        ventas = leer("ventas")
        oil = leer("oil")
        holidays = leer("holidays_events")
        stores = leer("stores")

        cutoff = pd.Timestamp(fecha_str)
        dataset = construir_dataset(ventas, oil, holidays, stores, cutoff)

        # Guardar con el cutoff en el nombre (idempotente)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        ruta_salida = DATA_PROCESSED / f"features_{fecha_str}.parquet"
        dataset.to_parquet(ruta_salida, index=False)
        return f"Features guardadas: {len(dataset)} filas → {ruta_salida}"

    from airflow.operators.python import get_current_context

    @task()
    def obtener_fecha() -> str:
        ctx = get_current_context()
        return ctx["ds"]

    fecha = obtener_fecha()
    construir_features(fecha)


preparacion_datos()
