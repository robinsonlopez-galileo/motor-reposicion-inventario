# Motor de Reposición de Inventario

> Pipeline MLOps que transforma ventas diarias de Corporación Favorita en recomendaciones semanales de reposición de inventario.

**Alumno:** Robinson López Hidalgo | Carnet: 26007448  
**Maestría en Data Science** — Universidad Galileo  
**Laboratorio 1 — Product Development**

---

## Arquitectura

```
Kaggle CSV
    │
    ▼
scripts/seed_historical.py
    │
    ├─ data/raw/          ← histórico hasta 2016-12-31  (DVC)
    └─ data/inbox/        ← archivos diarios 2017+       (DVC)
           │
           ▼
    DAG: ingesta_fuentes  (@daily)
           │
           ▼
    data/raw/*.parquet    (validado + consolidado)
           │
           ▼
    DAG: preparacion_datos  (@daily)
           │
           ▼
    data/processed/features_YYYY-MM-DD.parquet
           │
           ▼
    DAG: pronostico_semanal  (@weekly)
           │  entrenamiento + selección de modelo
           ▼
    data/forecasts/pronostico_YYYY-MM-DD.parquet
           │
           ▼
    DAG: seguimiento_desempeno  (@daily)
           │
           ▼
    data/reports/  (métricas de desempeño)
```

## Requisitos previos

- **Docker Desktop** ≥ 24 (con Docker Compose v2)
- **Python** ≥ 3.11 (solo para el script de seed)
- **DVC** ≥ 3.50 con soporte Google Drive: `pip install "dvc[gdrive]"`
- Cuenta de Google con acceso a Google Drive
- Datos de Kaggle: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## Reproducción completa desde cero

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/robinsonlopez-galileo/motor-reposicion-inventario.git
cd motor-reposicion-inventario
```

### Paso 2 — Descargar los datos de Kaggle

Descarga el archivo ZIP del siguiente enlace y extrae su contenido en una carpeta local:

```
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
```

Los archivos que necesitas (dentro del ZIP) son:
- `train.csv`
- `stores.csv`
- `holidays_events.csv`
- `oil.csv`
- `transactions.csv`

Guárdalos en una carpeta, por ejemplo: `~/kaggle_data/`

### Paso 3 — Configurar el remoto DVC (Google Drive)

**3a. Crear una carpeta en Google Drive**

1. Ve a [drive.google.com](https://drive.google.com)
2. Crea una carpeta llamada `motor-reposicion-dvc`
3. Copia el **ID de la carpeta** desde la URL: `https://drive.google.com/drive/folders/<FOLDER_ID>`

**3b. Configurar el remoto en DVC**

```bash
dvc remote modify myremote url gdrive://<FOLDER_ID>
```

O edita directamente `.dvc/config`:

```ini
[core]
    remote = myremote
[remote "myremote"]
    url = gdrive://<TU_FOLDER_ID_AQUI>
```

**3c. Primera autenticación**

La primera vez que corras `dvc push` o `dvc pull`, DVC abrirá el navegador para pedir acceso OAuth a tu Google Drive. Acepta y el token quedará guardado en `~/.cache/pydrive2fs/`.

### Paso 4 — Construir la carga histórica (seed)

```bash
pip install pandas pyarrow  # dependencias mínimas para el script
python scripts/seed_historical.py --kaggle-dir ~/kaggle_data
```

Esto tarda ~2 minutos y crea:
- `data/raw/` — histórico hasta 2016-12-31 en formato parquet
- `data/inbox/` — archivos diarios desde 2017-01-01 (simulando el sistema de origen)

### Paso 5 — Versionar los datos con DVC

```bash
dvc add data/raw data/inbox
git add data/raw.dvc data/inbox.dvc data/.gitignore
git commit -m "chore: add initial DVC-tracked data"
dvc push
```

### Paso 6 — Iniciar Airflow con Docker Compose

```bash
# Primera vez: inicializar la base de datos
docker compose up airflow-init

# Levantar todos los servicios
docker compose up -d
```

Espera ~60 segundos y luego abre: **http://localhost:8080**

- Usuario: `airflow`
- Contraseña: `airflow`

### Paso 7 — Activar los DAGs y hacer backfill

En la UI de Airflow, activa (toggle ON) los siguientes DAGs en este orden:

1. `ingesta_fuentes`
2. `preparacion_datos`
3. `pronostico_semanal`
4. `seguimiento_desempeno`

Para hacer backfill desde 2017-01-01:

```bash
docker compose exec airflow-scheduler \
  airflow dags backfill ingesta_fuentes \
  --start-date 2017-01-01 --end-date 2017-03-31

docker compose exec airflow-scheduler \
  airflow dags backfill preparacion_datos \
  --start-date 2017-01-01 --end-date 2017-03-31
```

---

## Reproducir un pronóstico pasado con DVC

Para demostrar la reproducibilidad, aquí se muestra cómo obtener exactamente el mismo pronóstico del **4 de febrero de 2017**:

### 1. Ver el historial de versiones del dataset

```bash
dvc log data/raw
```

Ejemplo de salida:
```
commit abc1234 (HEAD -> main)
Date: 2017-02-05
    chore: seed data up to 2017-02-04

commit def5678
Date: 2017-01-28
    chore: seed data up to 2017-01-28
```

### 2. Restaurar el estado de datos de esa fecha

```bash
# Cambiar al commit de Git correspondiente a los datos disponibles el 2017-02-04
git checkout def5678
dvc checkout
```

### 3. Lanzar el pronóstico

```bash
docker compose exec airflow-scheduler \
  airflow tasks test pronostico_semanal publicar_pronostico 2017-02-04
```

### 4. Comparar el resultado

```bash
# El pronóstico generado estará en:
ls data/forecasts/pronostico_2017-02-04.parquet

# Para comparar con el original versionado:
python -c "
import pandas as pd
original = pd.read_parquet('data/forecasts/pronostico_2017-02-04.parquet')
print(original.head(10))
print('Filas:', len(original))
"
```

**Resultado esperado** (reproduccible determinísticamente):
- Pronóstico de ventas para 1,782 combinaciones (54 tiendas × 33 familias)
- Horizonte de 16 días desde el 2017-02-04
- Columnas: `date`, `store_nbr`, `family`, `sales_pred`, `pedido_sugerido`

---

## Estructura del proyecto

```
motor-reposicion-inventario/
├── config/
│   ├── business_costs.yaml     # costos de stockout/merma, horizonte, cuantiles
│   └── model_params.yaml       # parámetros LightGBM, ventana de entrenamiento
├── dags/
│   ├── dag_ingesta.py          # @daily: valida y consolida la bandeja
│   ├── dag_preparacion.py      # @daily: construye features hasta el día actual
│   ├── dag_pronostico.py       # @weekly: entrena, selecciona modelo, publica
│   └── dag_seguimiento.py      # @daily: mide desempeño vs. ventas reales
├── data/
│   ├── inbox/                  # archivos entrantes (DVC)
│   ├── raw/                    # histórico consolidado (DVC)
│   ├── processed/              # features por fecha (DVC)
│   ├── forecasts/              # pronósticos (DVC)
│   └── reports/                # métricas de seguimiento (DVC)
├── models/                     # modelos entrenados (DVC)
├── scripts/
│   └── seed_historical.py      # construye raw/ e inbox/ desde los CSV de Kaggle
├── src/
│   ├── features/constructor.py # ingeniería de features, anti-leakage
│   ├── ingesta/
│   │   ├── lector.py           # lectura y consolidación de archivos
│   │   └── validadores.py      # validación de esquemas y datos
│   ├── modelo/entrenador.py    # baseline + LightGBM, métricas, criterio de reemplazo
│   ├── pedido/calculador.py    # newsvendor, pedido sugerido
│   └── seguimiento/evaluador.py # comparación pronóstico vs. real
├── tests/                      # pytest: 22 tests sobre src/
├── .dvc/config                 # remoto Google Drive
├── .github/workflows/ci.yml    # lint + tests en cada push
├── CLAUDE.md                   # reglas del proyecto para Claude
├── DECISIONES.md               # registro de decisiones técnicas
├── docker-compose.yml          # Airflow 3 + Postgres + Redis
└── pyproject.toml              # dependencias y configuración del paquete
```

---

## Correr los tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src
```

Salida esperada (22 tests):
```
tests/test_validadores.py  ......   6 passed
tests/test_features.py     .....    5 passed
tests/test_modelo.py       .......  7 passed
tests/test_pedido.py       ....     4 passed
```

---

## Configuración de parámetros

### Costos de negocio (`config/business_costs.yaml`)

```yaml
stockout_cost_per_unit: 0.30   # costo por unidad que faltó
waste_cost_per_unit: 0.10      # costo por unidad que se desperdició
forecast_horizon_days: 16      # días a futuro del pronóstico
order_quantile: 0.70           # cuantil newsvendor (q* = 0.3/(0.3+0.1) = 0.75)
```

### Modelo (`config/model_params.yaml`)

```yaml
lgbm:
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 63
training_window_days: 365
replacement_threshold_pct: 2.0  # RMSE debe mejorar ≥ 2% para reemplazar el modelo
```

---

## Decisiones de diseño

Ver [DECISIONES.md](DECISIONES.md) para el registro completo de decisiones técnicas, alternativas consideradas y justificaciones.

---

## Datos de Kaggle

Los datos provienen de la competencia **Store Sales - Time Series Forecasting** de Kaggle, que contiene ventas reales de Corporación Favorita (Ecuador), una cadena de supermercados. El conjunto incluye 3M de registros de ventas diarias para 54 tiendas y 33 categorías de productos entre 2013 y 2017.

> **Nota:** Los datos originales no están incluidos en este repositorio (están en `.gitignore` y controlados por DVC). Descárgalos desde Kaggle como se indica en el Paso 2.
