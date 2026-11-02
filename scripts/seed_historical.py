"""
seed_historical.py — Construye la carga histórica inicial desde los archivos de Kaggle.

Uso:
    python scripts/seed_historical.py --kaggle-dir /ruta/a/kaggle_data

Resultado:
    data/raw/ventas.parquet        (train.csv hasta 2016-12-31)
    data/raw/stores.parquet
    data/raw/holidays_events.parquet
    data/raw/oil.parquet
    data/raw/transactions.parquet
    data/inbox/ventas/ventas_YYYY-MM-DD.csv  (2017-01-01 en adelante)
    data/inbox/oil/oil_YYYY-MM-DD.csv
    data/inbox/transactions/transactions_YYYY-MM-DD.csv

La bandeja simula el sistema de origen: archivos que llegan periódicamente,
uno por fecha y por fuente.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INBOX = PROJECT_ROOT / "data" / "inbox"

# Fecha de corte: hasta aquí es "histórico"; desde el día siguiente es "bandeja"
CORTE_HISTORICO = pd.Timestamp("2016-12-31")


def seed_stores(kaggle_dir: Path) -> None:
    """Tiendas: dato estático, va directo a raw."""
    ruta = kaggle_dir / "stores.csv"
    df = pd.read_csv(ruta)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_RAW / "stores.parquet", index=False)
    logger.info("stores.parquet: %d tiendas", len(df))


def seed_holidays(kaggle_dir: Path) -> None:
    """Feriados: dato estático, va directo a raw."""
    ruta = kaggle_dir / "holidays_events.csv"
    df = pd.read_csv(ruta)
    df["date"] = pd.to_datetime(df["date"])
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_RAW / "holidays_events.parquet", index=False)

    # También va a la bandeja como archivo plano (por fuente)
    inbox_dir = DATA_INBOX / "holidays_events"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    bandeja = df[df["date"] > CORTE_HISTORICO]
    for fecha, grupo in bandeja.groupby("date"):
        fecha_str = fecha.strftime("%Y-%m-%d")
        grupo.to_csv(inbox_dir / f"holidays_events_{fecha_str}.csv", index=False)
    logger.info("holidays: %d filas raw + %d fechas en bandeja", len(df), bandeja["date"].nunique())


def seed_oil(kaggle_dir: Path) -> None:
    """Petróleo: histórico a raw, desde 2017-01-01 uno por día en bandeja."""
    df = pd.read_csv(kaggle_dir / "oil.csv")
    df["date"] = pd.to_datetime(df["date"])

    historico = df[df["date"] <= CORTE_HISTORICO]
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    historico.to_parquet(DATA_RAW / "oil.parquet", index=False)

    bandeja = df[df["date"] > CORTE_HISTORICO]
    inbox_dir = DATA_INBOX / "oil"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for fecha, grupo in bandeja.groupby("date"):
        fecha_str = fecha.strftime("%Y-%m-%d")
        grupo.to_csv(inbox_dir / f"oil_{fecha_str}.csv", index=False)
    logger.info("oil: %d filas raw + %d días en bandeja", len(historico), len(bandeja))


def seed_transactions(kaggle_dir: Path) -> None:
    """Transacciones: histórico a raw, desde 2017-01-01 en bandeja."""
    df = pd.read_csv(kaggle_dir / "transactions.csv")
    df["date"] = pd.to_datetime(df["date"])

    historico = df[df["date"] <= CORTE_HISTORICO]
    historico.to_parquet(DATA_RAW / "transactions.parquet", index=False)

    bandeja = df[df["date"] > CORTE_HISTORICO]
    inbox_dir = DATA_INBOX / "transactions"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for fecha, grupo in bandeja.groupby("date"):
        fecha_str = fecha.strftime("%Y-%m-%d")
        grupo.to_csv(inbox_dir / f"transactions_{fecha_str}.csv", index=False)
    logger.info("transactions: %d filas raw + %d días en bandeja", len(historico), len(bandeja))


def seed_ventas(kaggle_dir: Path) -> None:
    """
    Ventas (train.csv): histórico a raw, desde 2017-01-01 en bandeja.

    El archivo train.csv tiene 3M de filas; se procesa en chunks para eficiencia.
    """
    logger.info("Procesando train.csv (puede tardar ~30 segundos)...")
    chunks_historico = []
    inbox_dir = DATA_INBOX / "ventas"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    inbox_buffer: dict[str, list] = {}

    for chunk in pd.read_csv(kaggle_dir / "train.csv", chunksize=100_000, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"])
        historico_chunk = chunk[chunk["date"] <= CORTE_HISTORICO]
        bandeja_chunk = chunk[chunk["date"] > CORTE_HISTORICO]

        if not historico_chunk.empty:
            chunks_historico.append(historico_chunk)

        if not bandeja_chunk.empty:
            for fecha, grupo in bandeja_chunk.groupby("date"):
                fecha_str = fecha.strftime("%Y-%m-%d")
                if fecha_str not in inbox_buffer:
                    inbox_buffer[fecha_str] = []
                inbox_buffer[fecha_str].append(grupo)

    # Guardar histórico
    historico = pd.concat(chunks_historico, ignore_index=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    historico.to_parquet(DATA_RAW / "ventas.parquet", index=False)
    logger.info("ventas raw: %d filas hasta %s", len(historico), CORTE_HISTORICO.date())

    # Guardar bandeja
    for fecha_str, partes in inbox_buffer.items():
        df_dia = pd.concat(partes, ignore_index=True)
        df_dia.to_csv(inbox_dir / f"ventas_{fecha_str}.csv", index=False)
    logger.info("ventas bandeja: %d días (2017-01-01 en adelante)", len(inbox_buffer))


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye la carga histórica y la bandeja.")
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        required=True,
        help="Directorio con los CSV de Kaggle (train.csv, stores.csv, etc.)",
    )
    args = parser.parse_args()

    kaggle_dir = args.kaggle_dir.resolve()
    if not kaggle_dir.exists():
        raise FileNotFoundError(f"Directorio Kaggle no encontrado: {kaggle_dir}")

    logger.info("Iniciando seed histórico desde: %s", kaggle_dir)
    logger.info("Corte histórico: %s", CORTE_HISTORICO.date())

    seed_stores(kaggle_dir)
    seed_holidays(kaggle_dir)
    seed_oil(kaggle_dir)
    seed_transactions(kaggle_dir)
    seed_ventas(kaggle_dir)

    logger.info("✓ Seed completado. data/raw/ y data/inbox/ listos.")


if __name__ == "__main__":
    main()
