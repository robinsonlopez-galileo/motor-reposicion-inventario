"""
DAG de seguimiento del desempeño.

Cada vez que llegan ventas reales, compara los pronósticos publicados
para ese período y actualiza el historial de desempeño.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FORECASTS = PROJECT_ROOT / "data" / "forecasts"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

DEFAULT_ARGS = {
    "owner": "robinson.lopez",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="seguimiento_desempeno",
    description="Compara pronósticos publicados con ventas reales disponibles",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=True,
    default_args=DEFAULT_ARGS,
    tags=["seguimiento", "evaluacion"],
)
def seguimiento_desempeno():

    @task()
    def evaluar_pronosticos(fecha_str: str) -> dict:
        """
        Para la fecha dada, busca todos los pronósticos que incluyan esa fecha
        y los compara con las ventas reales observadas.
        """
        import sys
        import pandas as pd
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from seguimiento.evaluador import (
            cargar_pronostico, comparar_con_reales, resumen_desempeno, guardar_reporte
        )

        # Cargar ventas reales
        ruta_ventas = DATA_RAW / "ventas.parquet"
        if not ruta_ventas.exists():
            return {"status": "sin datos reales"}

        ventas = pd.read_parquet(ruta_ventas)
        ventas["date"] = pd.to_datetime(ventas["date"])
        fecha_eval = pd.Timestamp(fecha_str)

        # Buscar todos los pronósticos que contienen la fecha evaluada
        resultados = []
        for ruta in sorted(DATA_FORECASTS.glob("pronostico_*.parquet")):
            try:
                pron = cargar_pronostico(ruta)
                # Solo si este pronóstico cubre la fecha de evaluación
                if fecha_eval not in pron["date"].values:
                    continue
                comparacion = comparar_con_reales(pron, ventas)
                if comparacion.empty:
                    continue
                resumen = resumen_desempeno(comparacion)
                # Extraer cutoff_date del nombre del archivo
                cutoff_str = ruta.stem.replace("pronostico_", "")
                cutoff = pd.Timestamp(cutoff_str)
                guardar_reporte(comparacion, resumen, REPORTS_DIR, cutoff)
                resultados.append({"cutoff": cutoff_str, **resumen})
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Error evaluando %s: %s", ruta, exc)

        return {"evaluados": len(resultados), "fecha": fecha_str}

    from airflow.operators.python import get_current_context

    @task()
    def obtener_fecha() -> str:
        ctx = get_current_context()
        return ctx["ds"]

    fecha = obtener_fecha()
    evaluar_pronosticos(fecha)


seguimiento_desempeno()
