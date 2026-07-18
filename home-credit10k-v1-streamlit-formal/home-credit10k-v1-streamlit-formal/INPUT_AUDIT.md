# Revisión de v2.5 y entradas

La base v2.5 contenía un pipeline orientado a auditoría productiva con XGBoost, regresión logística, árbol, SHAP, LIME, fairness, auditoría temporal, manifests, scanner de privacidad y veredictos de benchmark. Para la School Edition se conservaron los principios útiles de split temporal, métricas AUC/Gini, estabilidad, controles negativos, explicabilidad y diagnóstico por brechas.

Se retiraron XGBoost, árbol principal, submissions, LIME obligatorio, fairness gobernante, fingerprints y veredictos complejos. CatBoost pasó a ser el modelo fuerte y la regresión logística el baseline.

El ZIP de datos contiene exclusivamente tablas `train` filtradas por el mismo conjunto de 10,000 `case_id`; no contiene directorio test. La ruta aceptada por el CLI es `csv_files` o `csv_files/train`.
