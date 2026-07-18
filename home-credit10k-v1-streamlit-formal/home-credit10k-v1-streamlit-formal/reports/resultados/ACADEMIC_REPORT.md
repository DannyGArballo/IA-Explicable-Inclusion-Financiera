# Home Credit 10K: Informe

## 1. Descripción del problema
Se estima el riesgo de incumplimiento a partir de la información disponible para solicitudes de crédito.

## 2. Arquitectura
El flujo de trabajo carga la tabla base, integra tablas de profundidad 0, resume tablas históricas por `case_id`, selecciona variables usando exclusivamente el conjunto de entrenamiento, divide los datos por `WEEK_NUM`, entrena CatBoost y una regresión logística, y calcula controles, métricas y explicaciones.

## 3. Preparación de datos
Se analizaron 10,000 solicitudes, 314 positivas y 92 semanas. Se excluyeron identificadores directos y variables temporales no usadas como predictores. Las tablas históricas se resumieron principalmente mediante conteo, mínimo, máximo, media, desviación estándar, primer valor, último valor y rango.

## 4. Partición temporal
El entrenamiento termina en la semana 51 y la validación en la semana 67. No existe solapamiento de `case_id` entre las particiones.

## 5. Modelos
CatBoost es el modelo principal; utiliza variables categóricas nativas y parada temprana. La regresión logística emplea imputación, estandarización, codificación indicadora y regularización L2.

## 6. Métricas
| Modelo | Partición | AUC | Gini | Pérdida logarítmica | Brier |
|---|---|---:|---:|---:|---:|
| CatBoost | entrenamiento | 0.9296 | 0.8591 | 0.4536 | 0.1422 |
| CatBoost | validación | 0.7048 | 0.4096 | 0.4515 | 0.1417 |
| CatBoost | prueba | 0.7961 | 0.5921 | 0.4471 | 0.1399 |
| Regresión logística | entrenamiento | 0.8939 | 0.7878 | 0.4489 | 0.1472 |
| Regresión logística | validación | 0.5317 | 0.0635 | 1.1785 | 0.2012 |
| Regresión logística | prueba | 0.6504 | 0.3008 | 0.5182 | 0.1443 |


## 7. Diagnóstico de sobreajuste
Brecha de Gini entre entrenamiento y prueba de CatBoost: 0.2670. Brecha de la regresión logística: 0.4870. La brecha de CatBoost requiere cautela. La incertidumbre es relevante porque solo hay 314 casos positivos en toda la muestra.

## 9. Importancia de variables
Se generan la importancia global de CatBoost, un resumen SHAP reducido sin `case_id` y los coeficientes de la regresión logística. Las explicaciones son agregadas y no contienen probabilidades individuales. Los identificadores técnicos de las variables se conservan en su forma original para mantener la trazabilidad con el código y los datos.

## 10. Controles
La probabilidad constante produce un Gini de 0 por construcción. La permutación aleatoria del objetivo obtuvo un Gini medio en prueba de -0.1443; estado: correcto.

## 11. Limitaciones
La base es una muestra académica, la tasa positiva es baja, las últimas semanas contienen pocos casos, la selección de tablas históricas es deliberadamente ligera y los intervalos obtenidos mediante remuestreo bootstrap no sustituyen una validación externa. La equidad no pudo evaluarse de forma concluyente y no se utilizó la base completa.

## 12. Conclusiones
El proyecto permite estudiar rendimiento, estabilidad temporal, explicabilidad y sobreajuste en todo tipo de equipo.

