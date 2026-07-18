# Validación de la aplicación Streamlit

## Alcance

Verificación de la interfaz interactiva, el artefacto CatBoost serializado y la generación de explicaciones SHAP.

## Entorno

- Python: 3.13.5
- Streamlit: 1.59.2
- CatBoost: 1.2.8
- pandas: 2.2.3
- NumPy: 2.3.5
- scikit-learn: 1.8.0

## Artefacto de inferencia

- Modelo: `streamlit_app/artifacts/catboost_model.cbm`
- Configuración: principal
- Semilla: 42
- Variables seleccionadas: 250
- Variables editables: 12
- Mejor iteración: 41
- Umbral inicial: 0.361068, estimado sobre validación mediante Youden J

## Pruebas realizadas

```text
python -m compileall -q streamlit_app.py streamlit_app
Resultado: correcto

pytest -q tests/test_streamlit_inference.py
Resultado: 3 passed

Streamlit AppTest
Excepciones de renderizado: 0
Controles deslizables: 13
Pestañas: 5
```

## Escenarios de control

```text
Perfil típico:                    0.269342
Ejemplo de menor puntaje:         0.128218
Ejemplo de mayor puntaje:         0.481553
```

Los escenarios generan valores válidos en el intervalo [0, 1] y producen salidas diferentes al modificar los controles.
