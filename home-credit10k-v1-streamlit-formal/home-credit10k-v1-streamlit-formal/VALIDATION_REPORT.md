# Validación de entrega

## Alcance

Esta entrega es una edición académica ligera construida sobre los principios útiles de v2.5. No continúa la auditoría productiva ni afirma rendimiento sobre la base completa.

> Los resultados corresponden a una muestra escolar de 10,000 solicitudes y no representan una validación productiva.

## Entradas verificadas

- `home-credit-explainable-ai-audited-v2_5(1).zip`: SHA-256 `ea8f7724088afda75353185f281417bac1edc8fff50708a1e3761a969418c052`; ZIP íntegro.
- `home-credit10k.zip`: SHA-256 `5ca3c79f2caffb705d959133dc410092659bf17c9b1a5e5641fdad88296dafc0`; ZIP íntegro.
- Prompt Markdown: SHA-256 `990c8796d481d99c1a2ed5518b0d857c7fae9e5f3f3ca4441c704f627d6d7be5`.
- Ruta correcta de datos: `home-credit-school-10k/csv_files`.

## Dataset escolar

- Casos: 10,000
- Positivos: 314
- Negativos: 9,686
- Semanas: 92 (0–91)
- Tasa positiva: 3.1400%

## Feature engineering y split

- Features iniciales: 516
- Features finales: 250
- Train: 6,997 casos, 215 positivos, semanas 0–51
- Validation: 1,501 casos, 60 positivos, semanas 52–67
- Test: 1,502 casos, 39 positivos, semanas 68–91
- Solapamiento de `case_id`: False

## Métricas school

| Modelo | Split | AUC | Gini | Log-loss | Brier |
|---|---|---:|---:|---:|---:|
| CatBoost | train | 0.929556 | 0.859113 | 0.453619 | 0.142183 |
| CatBoost | validation | 0.704788 | 0.409577 | 0.451496 | 0.141718 |
| CatBoost | test | 0.796072 | 0.592145 | 0.447069 | 0.139861 |
| Logística | train | 0.893924 | 0.787849 | 0.448882 | 0.147152 |
| Logística | validation | 0.531726 | 0.063451 | 1.178505 | 0.201212 |
| Logística | test | 0.650420 | 0.300840 | 0.518244 | 0.144258 |

## Brechas de generalización

- CatBoost train-validation Gini: 0.449536
- CatBoost train-test Gini: 0.266968
- Logística train-validation Gini: 0.724398
- Logística train-test Gini: 0.487009

Las brechas son materialmente positivas, especialmente en logística. Por ello el reporte no interpreta un Gini alto de train como prueba de buen desempeño y marca cautela por overfitting y por el bajo número de positivos.

## Controles

- Probabilidad constante: Gini 0.0 en train, validation y test.
- Target shuffle school: Gini medio test -0.144315; estado `ok` con 10 repeticiones.
- Target shuffle smoke: estado `warning`; su advertencia se conserva porque el test smoke contiene muy pocos positivos y el control es inestable.

## Duraciones

- Smoke pipeline: 8.48 s.
- School pipeline: 22.77 s.
- Máximo RSS observado con `/usr/bin/time`: aproximadamente 1.45 GB smoke y 1.75 GB school en este entorno.

## Entorno probado

- Python: 3.13.5 (main, May  5 2026, 21:05:52) [GCC 14.2.0]
- Dependencias: pandas 2.2.3, numpy 2.3.5, scikit-learn 1.8.0, catboost 1.2.8, matplotlib 3.10.8, psutil 7.2.2, pytest 9.0.2
- Threads: 2

## Resultado

- `compileall`: ejecutado correctamente.
- `pytest`: 10 passed, 0 failed, 0 skipped.
- Perfil smoke: ejecutado correctamente.
- Perfil school: ejecutado correctamente.
- Ensemble: implementado como demostración opcional, no requerido para la validación principal.

## Limitaciones

La muestra tiene solo 314 positivos; los splits tardíos son pequeños; la validación y el test presentan diferencias temporales; las explicaciones SHAP se limitan a 200 observaciones; el feature engineering histórico es deliberadamente compacto; no hay validación externa, fairness concluyente ni afirmación productiva. No se incluyen datos, probabilidades individuales, `case_id` reales ni explicaciones por persona en el paquete.
