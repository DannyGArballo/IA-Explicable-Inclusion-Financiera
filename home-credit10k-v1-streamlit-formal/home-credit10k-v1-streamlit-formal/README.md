# Home Credit 10K · Evaluación interactiva

Aplicación de Streamlit para explorar un modelo CatBoost de riesgo crediticio con 10,000 solicitudes. La interfaz permite modificar variables relevantes, consultar el índice generado por el modelo, ajustar el umbral de clasificación y revisar una explicación SHAP individual.

## Inicio rápido

```powershell
python -m pip install -r requirements.txt
python -m streamlit run .\streamlit_app.py
```

También puede iniciarse con:

- `run_streamlit.bat`
- `run_streamlit.ps1`

## Contenido de la interfaz

- Evaluación interactiva con doce variables editables.
- Escenarios de referencia: típico, menor índice y mayor índice.
- Umbral de clasificación ajustable.
- Aprobación preliminar, revisión adicional o no aprobación.
- Explicación SHAP para cada resultado.
- Métricas AUC y Gini.
- Gráficos de importancia global y SHAP.
- Resumen del conjunto de datos y de la partición temporal.
- Diccionario ampliado de variables con origen, transformación y percentiles de referencia.

## Archivos principales

```text
streamlit_app.py
streamlit_app/
├── inference.py
├── MODEL_CARD.md
└── artifacts/
    ├── catboost_model.cbm
    ├── model_metadata.json
    └── selected_features_training.csv
```

## Dependencias

Las versiones compatibles se encuentran en:

```text
requirements.txt
requirements.lock.txt
```

## Documentación

- `INSTRUCCIONES_STREAMLIT.md`: instalación y ejecución.
- `streamlit_app/MODEL_CARD.md`: configuración del modelo y artefactos.
- `STREAMLIT_VALIDATION.md`: pruebas realizadas sobre la interfaz.

El código de preparación, entrenamiento y generación de reportes se conserva en las carpetas y archivos originales del proyecto.
