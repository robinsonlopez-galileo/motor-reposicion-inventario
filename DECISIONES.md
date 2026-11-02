# DECISIONES.md — Registro de Decisiones Técnicas y de Producto

## Organización del documento
Cada decisión incluye: contexto, alternativas consideradas y justificación de la elección.

---

## D1 — Fecha de corte para la carga histórica

**Decisión:** Los datos hasta el **31 de diciembre de 2016** forman el histórico inicial (`data/raw/`). Los datos de 2017 (enero–agosto, que es hasta donde llega `train.csv`) se colocan en la bandeja (`data/inbox/`) y se procesan por fecha como si llegaran del sistema de origen.

**Alternativas consideradas:**
- Usar todo `train.csv` como histórico → eliminaría la simulación de ingesta incremental
- Corte en 2015 → poco histórico para construir lags de 28 días con buen cubrimiento

**Justificación:** 2013–2016 da ~4 años de histórico estacional, suficiente para lags y medias móviles. El período 2017 en la bandeja permite validar el funcionamiento del pipeline de ingesta con datos reales.

---

## D2 — Disponibilidad temporal de la información (no data leakage)

**Decisión:** Toda predicción usa únicamente información disponible en el momento del corte (`cutoff_date`). La función `filtrar_hasta_corte()` en `src/features/constructor.py` es el punto único de control.

**Alternativas consideradas:**
- Validación aleatoria (train_test_split con shuffle) → estándar en muchos tutoriales pero invalida los resultados en series temporales
- Incluir variables de la misma fecha (ej. precio del petróleo del día a pronosticar) → viola el principio si ese dato no estaba disponible al emitir el pronóstico

**Justificación:** El enunciado lo establece como principio fundamental. Los lags se calculan con `.shift(1)` mínimo para evitar usar el valor del día actual al predecirlo.

---

## D3 — Manejo de correcciones (datos que se re-entregan)

**Decisión:** `consolidar_nuevos_registros()` en `src/ingesta/lector.py` usa `drop_duplicates(keep='last')`. Los registros más recientes en el orden de procesamiento reemplazan a los anteriores con las mismas claves.

**Alternativas consideradas:**
- Mantener ambas versiones con timestamp de ingesta → mayor complejidad, no pedido por el enunciado
- Ignorar correcciones (keep='first') → acumularía datos incorrectos en el histórico

**Justificación:** El correo del 27 de octubre indica que "algunos datos pueden corregirse después de publicados". El modelo last-write-wins es simple, auditable y correcto para este caso.

---

## D4 — Modelo baseline vs. modelo ML

**Decisión:**
- **Baseline:** Media móvil de las últimas 8 semanas para el mismo día de la semana, por (tienda, familia). Sin entrenamiento; puramente estadístico.
- **Modelo ML:** LightGBM regressor con features de lags (7, 14, 28 días), medias móviles (7, 28, 84 días), variables temporales y features externas (petróleo, feriados, tipo de tienda).

**Alternativas consideradas:**
- Prophet (Facebook) → bueno para una serie, pero no escala a 54 tiendas × 33 familias = 1,782 series de manera eficiente
- SARIMA por serie → mismo problema de escala
- XGBoost vs. LightGBM → LightGBM es más rápido en datasets grandes y maneja categorical features nativamente

**Justificación:** LightGBM es el estándar en competencias de forecasting tabular (ganó Corporación Favorita en Kaggle). El baseline de media móvil es interpretable y reproduce el comportamiento actual del equipo de operaciones (pedidos por criterio manual).

---

## D5 — Validación temporal

**Decisión:** Se usa validación temporal con ventana fija: las últimas **4 semanas** antes del cutoff son el set de validación; el resto es entrenamiento. No se hace k-fold.

**Alternativas consideradas:**
- k-fold cronológico (expanding window) → más robusto pero requiere entrenar k modelos; el costo es alto a 1,782 series
- Validación aleatoria → invalida resultados en series temporales (Lessmann et al., 2015 advierten sobre esto)

**Justificación:** La ventana de 4 semanas reproduce el horizonte real de uso del modelo y es coherente con el horizonte de pronóstico de 16 días.

---

## D6 — Criterio de reemplazo del modelo

**Decisión:** El modelo nuevo reemplaza al vigente solo si su **RMSE en validación mejora al menos 2%** respecto al modelo en producción.

**Alternativas consideradas:**
- Reemplazar siempre → riesgo de degradación por sobreajuste o datos corruptos
- Umbral basado en costo de negocio → más preciso pero requiere etiquetas de costo reales que aún no tenemos

**Justificación:** El 2% es un umbral conservador que evita cambios de modelo por ruido estadístico. El parámetro está en `config/model_params.yaml` y puede ajustarse sin tocar el código.

---

## D7 — Pedido sugerido: cuantil newsvendor

**Decisión:** El pedido se calcula con la fórmula del **newsvendor problem**: `q* = stockout_cost / (stockout_cost + waste_cost)`. Los costos están en `config/business_costs.yaml`.

**Alternativas consideradas:**
- Usar directamente la mediana del pronóstico → subestimaría el pedido cuando el costo de quiebre es mayor que el de merma
- Inventario de seguridad fijo (ej. +20%) → no considera la asimetría real de costos

**Justificación:** La teoría del newsvendor es el modelo estándar para decisiones de inventario bajo incertidumbre. Los costos son configurables sin modificar el código, lo que permite que operaciones los ajuste.

---

## D8 — DVC remoto: Google Drive

**Decisión:** Se usa **Google Drive** como remoto DVC.

**Alternativas consideradas:**
- DagsHub → muy bueno para ML (UI de experimentos, integración con MLflow), pero requiere cuenta separada y registro
- Amazon S3 / GCS → ideales para producción, pero requieren credenciales de nube de pago
- Carpeta local → no permite que el docente descargue los artefactos al clonar el repo

**Justificación:** Google Drive es gratuito, el alumno ya tiene cuenta con el correo de Galileo, y la integración con DVC (`dvc[gdrive]`) es estable. El único inconveniente es el paso de autenticación OAuth la primera vez, que se documenta en el README.

---

## D9 — Idempotencia de las tareas

**Decisión:** Todas las tareas son idempotentes: usan la fecha de ejecución (`ds`) en el nombre del archivo de salida. Ejecutar la misma tarea dos veces produce el mismo resultado y no acumula duplicados.

**Implementación:**
- Features: `data/processed/features_YYYY-MM-DD.parquet`
- Pronósticos: `data/forecasts/pronostico_YYYY-MM-DD.parquet`
- Ingesta: `drop_duplicates(keep='last')` al consolidar

**Justificación:** Requisito explícito del enunciado. Permite backfill sin efectos secundarios.
