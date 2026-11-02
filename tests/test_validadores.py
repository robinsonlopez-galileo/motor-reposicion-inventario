"""Tests para el módulo de validación de fuentes."""
import sys
from pathlib import Path
import tempfile
import pytest
import pandas as pd

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingesta.validadores import validar_columnas, validar_ventas, validar_archivo


def test_validar_columnas_ok():
    df = pd.DataFrame({
        "id": [1], "date": ["2017-01-01"], "store_nbr": [1],
        "family": ["BEVERAGES"], "sales": [10.0], "onpromotion": [0],
    })
    errores = validar_columnas(df, "train")
    assert errores == [], f"No debería haber errores, pero hubo: {errores}"


def test_validar_columnas_faltantes():
    df = pd.DataFrame({"id": [1], "date": ["2017-01-01"]})
    errores = validar_columnas(df, "train")
    assert len(errores) > 0
    assert "Columnas faltantes" in errores[0]


def test_validar_ventas_negativas():
    df = pd.DataFrame({
        "store_nbr": [1], "family": ["BEVERAGES"], "sales": [-5.0], "date": ["2017-01-01"]
    })
    errores = validar_ventas(df)
    assert any("negativas" in e for e in errores)


def test_validar_ventas_store_invalido():
    df = pd.DataFrame({
        "store_nbr": [999], "family": ["BEVERAGES"], "sales": [10.0], "date": ["2017-01-01"]
    })
    errores = validar_ventas(df)
    assert any("inválidos" in e for e in errores)


def test_validar_archivo_no_existe():
    resultado = validar_archivo(Path("/ruta/que/no/existe.csv"), "train")
    assert not resultado.valido
    assert "no encontrado" in resultado.errores[0].lower()


def test_validar_archivo_csv_correcto():
    contenido = "id,date,store_nbr,family,sales,onpromotion\n1,2017-01-01,1,BEVERAGES,10.0,0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(contenido)
        ruta = Path(f.name)
    try:
        resultado = validar_archivo(ruta, "train")
        assert resultado.valido, f"Debería ser válido: {resultado.errores}"
        assert resultado.filas == 1
    finally:
        ruta.unlink()
