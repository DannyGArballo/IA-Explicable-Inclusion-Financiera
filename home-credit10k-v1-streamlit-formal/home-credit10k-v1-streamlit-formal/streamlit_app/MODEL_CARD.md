# Ficha del modelo CatBoost 10K

## Propósito

Artefacto de inferencia utilizado por la interfaz de Streamlit para calcular un índice de riesgo y explicar cada resultado mediante SHAP.

## Configuración

- Algoritmo: CatBoost.
- Semilla: 42.
- Variables seleccionadas: 250.
- Partición temporal: entrenamiento, validación y prueba.
- Mejor iteración: registrada en `artifacts/model_metadata.json`.

## Archivos

- Modelo: `artifacts/catboost_model.cbm`.
- Metadatos y rangos de referencia: `artifacts/model_metadata.json`.
- Variables de entrenamiento utilizadas para construir controles: `artifacts/selected_features_training.csv`.

## Umbral

El valor inicial se calculó sobre validación mediante el estadístico de Youden J. La interfaz permite modificarlo para comparar distintos criterios de clasificación.

## Explicabilidad

La aplicación calcula valores SHAP para cada evaluación y presenta las variables con mayor contribución positiva o negativa.
