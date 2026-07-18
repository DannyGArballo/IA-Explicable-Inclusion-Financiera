# Home Credit — Edición Escolar 10K: Informe académico

> Los resultados corresponden a una muestra escolar y no representan una validación productiva.

## 1. Descripción del problema
Se estima el riesgo de incumplimiento a partir de la información disponible para solicitudes de crédito. El objetivo es didáctico: comparar un modelo fuerte con un modelo de referencia interpretable, sin afirmar validez productiva.

## 2. Arquitectura
El flujo de trabajo carga la tabla base, integra tablas de profundidad 0, resume tablas históricas por `case_id`, selecciona variables usando exclusivamente el conjunto de entrenamiento, divide los datos por `WEEK_NUM`, entrena CatBoost y una regresión logística, y calcula controles, métricas y explicaciones.

## 3. Preparación de datos
Se analizaron 1,000 solicitudes, 31 positivas y 90 semanas. Se excluyeron identificadores directos y variables temporales no usadas como predictores. Las tablas históricas se resumieron principalmente mediante conteo, mínimo, máximo, media, desviación estándar, primer valor, último valor y rango.

## 4. Partición temporal
El entrenamiento termina en la semana 51 y la validación en la semana 70. No existe solapamiento de `case_id` entre las particiones.

## 5. Modelos
CatBoost es el modelo principal; utiliza variables categóricas nativas y parada temprana. La regresión logística emplea imputación, estandarización, codificación indicadora y regularización L2.

## 6. Métricas
| Modelo | Partición | AUC | Gini | Pérdida logarítmica | Brier |
|---|---|---:|---:|---:|---:|
| CatBoost | entrenamiento | 0.8885 | 0.7769 | 0.6286 | 0.2179 |
| CatBoost | validación | 0.8350 | 0.6701 | 0.6377 | 0.2225 |
| CatBoost | prueba | 0.8187 | 0.6374 | 0.6296 | 0.2185 |
| Regresión logística | entrenamiento | 0.9970 | 0.9939 | 0.1364 | 0.0371 |
| Regresión logística | validación | 0.3350 | -0.3299 | 0.7637 | 0.1306 |
| Regresión logística | prueba | 0.7838 | 0.5676 | 0.3419 | 0.0950 |

## 7. Interpretación del Gini
Gini = 2 × AUC − 1. Un Gini alto en entrenamiento no demuestra por sí solo sobreajuste; debe compararse con validación y prueba, que representan semanas posteriores no observadas durante el entrenamiento.

## 8. Diagnóstico de sobreajuste
Brecha de Gini entre entrenamiento y prueba de CatBoost: 0.1395. Brecha de la regresión logística: 0.4264. La brecha de CatBoost no es grande en esta ejecución, aunque la muestra sigue siendo limitada. La incertidumbre es relevante porque solo hay 31 casos positivos en toda la muestra.

## 9. Importancia de variables
Se generan la importancia global de CatBoost, un resumen SHAP reducido sin `case_id` y los coeficientes de la regresión logística. Las explicaciones son agregadas y no contienen probabilidades individuales. Los identificadores técnicos de las variables se conservan en su forma original para mantener la trazabilidad con el código y los datos.

## 10. Controles
La probabilidad constante produce un Gini de 0 por construcción. La permutación aleatoria del objetivo obtuvo un Gini medio en prueba de -0.2808; estado: advertencia.

## 11. Limitaciones
La base es una muestra académica, la tasa positiva es baja, las últimas semanas contienen pocos casos, la selección de tablas históricas es deliberadamente ligera y los intervalos obtenidos mediante remuestreo bootstrap no sustituyen una validación externa. La equidad no pudo evaluarse de forma concluyente y no se utilizó la base completa.

## 12. Conclusiones
El proyecto permite estudiar rendimiento, estabilidad temporal, explicabilidad y sobreajuste en una computadora portátil común. Sus resultados sirven para aprendizaje y presentación escolar; no deben usarse para decisiones crediticias reales ni para una comparación directa con soluciones ganadoras de Kaggle.

**Declaración final:** Los resultados corresponden a una muestra escolar y no representan una validación productiva.
