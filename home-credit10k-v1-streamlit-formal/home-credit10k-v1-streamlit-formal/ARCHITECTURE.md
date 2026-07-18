# Arquitectura simplificada

La edición escolar reutiliza de v2.5 los principios de separación temporal, métricas de discriminación/calibración, controles negativos y diagnóstico por brechas. Se sustituyó XGBoost por CatBoost, se conservó regresión logística y se retiraron componentes orientados a auditoría productiva (submission, fairness gobernante, LIME obligatorio, fingerprints complejos y veredictos de benchmark).

El flujo principal es:

`train_base → feature engineering ligero → split temporal → selección train-only → CatBoost + logística → métricas/controles → SHAP/coefs → reporte`.
