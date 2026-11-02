"""
DAG de ingesta y validación de fuentes.

Responsabilidad: leer la bandeja, validar los archivos del día y consolidar
en data/raw/. No contiene lógica de negocio; toda la lógica está en src/ingesta/.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
INBOX = PROJECT_ROOT / "data" / "inbox"

FUENTES = ["ventas", "oil", "transactions", "holidays_events"]

DEFAULT_ARGS = {
    "owner": "robinson.lopez",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ingesta_fuentes",
    description="Lee la bandeja, valida y consolida en data/raw/",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=True,   # True para procesar períodos pasados (backfill)
    default_args=DEFAULT_ARGS,
    tags=["ingesta", "datos"],
)
def ingesta_fuentes():

    @task()
    def validar_bandeja(fuente: str, fecha_str: str) -> dict:
        """Valida el archivo de la bandeja para la fuente y fecha dadas."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from ingesta.validadores import validar_archivo

        ruta = INBOX / fuente / f"{fuente}_{fecha_str}.csv"
        resultado = validar_archivo(ruta, fuente)

        if not resultado.valido:
            raise ValueError(
                f"Validación fallida para {fuente}@{fecha_str}: {resultado.errores}"
            )
        return {"fuente": fuente, "fecha": fecha_str, "filas": resultado.filas}

    @task()
    def consolidar(fuente: str, fecha_str: str, info_validacion: dict) -> str:
        """Consolida el archivo validado en el histórico raw (parquet)."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from ingesta.lector import leer_bandeja, leer_historico, consolidar_nuevos_registros

        nuevos = leer_bandeja(INBOX, fuente, fecha_str)
        if nuevos is None:
            return f"Sin datos nuevos para {fuente}@{fecha_str}"

        ruta_historico = DATA_RAW / f"{fuente}.parquet"
        if ruta_historico.exists():
            import pandas as pd
            historico = pd.read_parquet(ruta_historico)
        else:
            import pandas as pd
            historico = pd.DataFrame()

        KEY_COLS = {
            "ventas": ["date", "store_nbr", "family"],
            "oil": ["date"],
            "transactions": ["date", "store_nbr"],
            "holidays_events": ["date", "description"],
        }

        consolidado = consolidar_nuevos_registros(
            historico, nuevos, KEY_COLS.get(fuente, ["date"])
        )
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        consolidado.to_parquet(ruta_historico, index=False)
        return f"Consolidado {fuente}: {len(consolidado)} filas totales"

    # Airflow pasa la fecha de ejecución; la exponemos como string
    from airflow.operators.python import get_current_context

    @task()
    def obtener_fecha() -> str:
        ctx = get_current_context()
        return ctx["ds"]  # formato YYYY-MM-DD

    fecha = obtener_fecha()

    for fuente in FUENTES:
        info = validar_bandeja.override(task_id=f"validar_{fuente}")(fuente, fecha)
        consolidar.override(task_id=f"consolidar_{fuente}")(fuente, fecha, info)


ingesta_fuentes()
