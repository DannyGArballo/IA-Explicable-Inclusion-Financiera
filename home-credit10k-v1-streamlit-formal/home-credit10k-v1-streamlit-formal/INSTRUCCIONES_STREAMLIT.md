# Aplicación interactiva de Streamlit

La aplicación utiliza el modelo CatBoost serializado en `streamlit_app/artifacts/` y puede ejecutarse sin volver a entrenar el pipeline.

## Windows PowerShell

```powershell
cd "RUTA\home-credit10k-v1-streamlit"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run .\streamlit_app.py
```

También puede iniciarse con doble clic en `run_streamlit.bat` después de instalar las dependencias.

## Contenido

- Evaluación interactiva con las variables SHAP más relevantes.
- Umbral de clasificación ajustable.
- Resultado: aprobación preliminar, revisión adicional o no aprobación.
- Explicación SHAP individual para cada combinación de valores.
- Métricas AUC/Gini y gráficos globales.
- Resumen del conjunto de datos y de la partición temporal.
- Diccionario ampliado de las doce variables editables.
