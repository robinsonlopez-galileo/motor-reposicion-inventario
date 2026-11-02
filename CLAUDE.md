# Motor de Reposición de Inventario — Reglas del Proyecto

## Contexto
Pipeline de ML orquestado con Airflow 3 que genera pronósticos de ventas y pedidos sugeridos
para Corporación Favorita. Datos de Kaggle "Store Sales – Time Series Forecasting".

## Estructura de directorios
```
src/               # Módulos Python testeables (sin dependencia de Airflow)
  ingesta/         # Lectura y validación de fuentes
  features/        # Ingeniería de características
  modelo/          # Entrenamiento, evaluación y selección de modelos
  pedido/          # Cálculo del pedido sugerido a partir del pronóstico
  seguimiento/     # Comparación pronóstico vs. ventas reales
dags/              # Solo orquestación; la lógica va en src/
tests/             # pytest; espejo de src/
config/            # YAML versionados con DVC
data/              # Nunca en Git; versionado con DVC
scripts/           # Bootstrap de datos (seed_historical, seed_inbox)
```

## Reglas de desarrollo
1. **Toda la lógica de negocio va en `src/`**, no en los DAGs.
2. **Los DAGs solo llaman funciones de `src/`**; no deben tener lógica propia.
3. **Tareas idempotentes**: ejecutar una tarea dos veces sobre el misma fecha produce el mismo resultado.
4. **Sin data leakage**: ninguna función puede usar información de fechas posteriores al `cutoff_date`.
5. **DVC para todo**: datos de entrada, procesados, modelos y pronósticos se rastrean con DVC.
6. **Pruebas primero**: toda función pública en `src/` debe tener al menos un test.
7. **YAML de configuración**: costos de negocio y parámetros del modelo en `config/`; nunca hardcodeados.

## Convenciones de código
- Nombres en snake_case; clases en PascalCase.
- Type hints en todas las funciones públicas.
- Docstrings en español.
- Máximo 100 caracteres por línea.

## Flujo de datos
```
Bandeja (inbox/) → ingesta → data/raw/ → features → data/processed/ 
→ modelo → models/ → pedido → data/forecasts/ → seguimiento → reports/
```

## Ramas y commits
- `main` protegida; todo cambio vía PR.
- Ramas: `feature/<descripcion>`, `fix/<descripcion>`.
- Commits en español, imperativo: "Agrega módulo de ingesta".
