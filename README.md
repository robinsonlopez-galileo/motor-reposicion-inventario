# Motor de Reposición de Inventario

> Pipeline MLOps que transforma ventas diarias de Corporación Favorita en recomendaciones semanales de reposición de inventario.

**Alumno:** Robinson López Hidalgo | Carnet: 26007448
**Maestría en Data Science** — Universidad Galileo
**Laboratorio 1 — Product Development**

---

## Arquitectura

```
Kaggle CSV
    |
    v
scripts/seed_historical.py
    |
    +- data/raw/          <- histórico hasta 2016-12-31  (Git LFS)
    +- data/inbox/        <- archivos diarios 2017+       (Git LFS)
           |
           v
    DAG: ingesta_fuentes  (@daily)
           |
           v
    data/raw/*.parquet    (validado + consolidado)
           |
           v
    DAG: preparacion_datos  (@daily)
           |
           v
    data/processed/features_YYYY-MM-DD.parquet
           |
           v
    DAG: pronostico_semanal  (@weekly)
           |  entrenamiento + selección de modelo
           v
    data/forecasts/pronostico_YYYY-MM-DD.parquet
           |
           v
    DAG: seguimiento_desempeno  (@daily)
           |
           v
    data/reports/  (métricas de desempeño)
```

## Requisitos previos

- **Docker Desktop** >= 24 (con Docker Compose v2)
- **Python** >= 3.11
- **Git LFS** instalado: https://git-lfs.com
- Los datos de Kaggle ya están incluidos en el repo via Git LFS

---

## Reproducción completa desde cero

### Paso 1 — Clonar el repositorio (incluye los datos)

```bash
git lfs install
git clone https://github.com/robinsonlopez-galileo/motor-reposicion-inventario.git
cd motor-reposicion-inventario
git lfs pull
```

Los datos de data/raw/ y data/inbox/ se descargan automáticamente con Git LFS.

### Paso 2 — Verificar los datos

```bash
python -c "
import pandas as pd
from pathlib import Path
for f in Path('data/raw').glob('*.parquet'):
    df = pd.read_parquet(f)
    print(f'{f.name}: {len(df):,} filas')
"
```

Salida esperada:
```
holidays_events.parquet: 350 filas
oil.parquet: 1,044 filas
stores.parquet: 54 filas
transactions.parquet: 71,391 filas
ventas.parquet: 2,596,374 filas
```

### Paso 3 — Correr los tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Salida esperada: **22 passed**

### Paso 4 — Iniciar Airflow con Docker Compose

```bash
docker compose up airflow-init
docker compose up -d
```

Abre http://localhost:8080 — Usuario: airflow / Contraseña: airflow

### Paso 5 — Activar los DAGs y hacer backfill

Activa en la UI en este orden:
1. ingesta_fuentes
2. preparacion_datos
3. pronostico_semanal
4. seguimiento_desempeno

```bash
docker compose exec airflow-scheduler airflow dags backfill ingesta_fuentes --start-date 2017-01-01 --end-date 2017-03-31
docker compose exec airflow-scheduler airflow dags backfill preparacion_datos --start-date 2017-01-01 --end-date 2017-03-31
```

---

## Estructura del proyecto

```
motor-reposicion-inventario/
+-- config/
|   +-- business_costs.yaml
|   +-- model_params.yaml
+-- dags/
|   +-- dag_ingesta.py
|   +-- dag_preparacion.py
|   +-- dag_pronostico.py
|   +-- dag_seguimiento.py
+-- data/
|   +-- inbox/         (Git LFS - archivos diarios 2017)
|   +-- raw/           (Git LFS - histórico parquet)
|   +-- processed/
|   +-- forecasts/
|   +-- reports/
+-- models/
+-- scripts/
|   +-- seed_historical.py
+-- src/
|   +-- features/constructor.py
|   +-- ingesta/lector.py
|   +-- ingesta/validadores.py
|   +-- modelo/entrenador.py
|   +-- pedido/calculador.py
|   +-- seguimiento/evaluador.py
+-- tests/
+-- .github/workflows/ci.yml
+-- CLAUDE.md
+-- DECISIONES.md
+-- docker-compose.yml
+-- pyproject.toml
```

---

## Resultados de tests

```
tests/test_validadores.py  ......   6 passed
tests/test_features.py     .....    5 passed
tests/test_modelo.py       .......  7 passed
tests/test_pedido.py       ....     4 passed
========================= 22 passed in 4.50s =========================
```

---

## Configuración de parámetros

```yaml
# config/business_costs.yaml
stockout_cost_per_unit: 0.30
waste_cost_per_unit: 0.10
forecast_horizon_days: 16
order_quantile: 0.70
```

```yaml
# config/model_params.yaml
lgbm:
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 63
training_window_days: 365
replacement_threshold_pct: 2.0
```

---

## Decisiones de diseño

Ver DECISIONES.md para el registro completo de 9 decisiones técnicas con alternativas y justificaciones.

---

## Datos

2,596,374 registros de ventas diarias para 54 tiendas y 33 familias de productos (2013-2017).
Fuente: Store Sales - Time Series Forecasting (Kaggle).
Versionados con Git LFS en este repositorio.
