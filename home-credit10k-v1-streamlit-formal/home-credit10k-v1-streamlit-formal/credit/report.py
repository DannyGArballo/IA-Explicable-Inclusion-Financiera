from __future__ import annotations
from pathlib import Path
import json

DISCLAIMER = "Los resultados corresponden a una muestra escolar de 10,000 solicitudes y no representan una validación productiva."


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def write_report(output_dir: Path, payload: dict) -> Path:
    cb = payload["metrics"]["catboost"]
    lg = payload["metrics"]["logistic"]
    ds = payload["dataset"]
    split = payload["split"]
    controls = payload["controls"]
    overfit_cb = cb["gaps"]["train_minus_test_gini"]
    overfit_lg = lg["gaps"]["train_minus_test_gini"]
    diagnosis = (
        "La brecha de CatBoost merece cautela." if overfit_cb > 0.15 else
        "La brecha de CatBoost no es grande para esta ejecución, aunque la muestra sigue siendo limitada."
    )
    lines = [
        "# Home Credit School Edition 10K — Reporte académico",
        "",
        f"> {DISCLAIMER}",
        "",
        "## 1. Descripción del problema",
        "Se estima la probabilidad de incumplimiento usando información disponible para solicitudes de crédito. El objetivo es didáctico: comparar un modelo fuerte con un baseline interpretable sin afirmar validez productiva.",
        "",
        "## 2. Arquitectura",
        "El pipeline carga la tabla base, integra tablas depth-0, resume tablas históricas por `case_id`, selecciona variables exclusivamente con train, divide por `WEEK_NUM`, entrena CatBoost y regresión logística, calcula controles, métricas y explicaciones.",
        "",
        "## 3. Preparación de datos",
        f"Se analizaron {ds['rows']:,} solicitudes, {ds['positives']} positivas y {ds['weeks']} semanas. Se excluyeron identificadores directos y variables temporales no usadas como predictores. Las tablas históricas se resumieron principalmente con count, min, max, mean, std, first, last y range.",
        "",
        "## 4. Split temporal",
        f"Train termina en la semana {split['train_end_week']} y validation en la semana {split['validation_end_week']}. No existe solapamiento de `case_id`.",
        "",
        "## 5. Modelos",
        "CatBoost es el modelo principal, con categóricas nativas y early stopping. La regresión logística usa imputación, estandarización, one-hot encoding y regularización L2.",
        "",
        "## 6. Métricas",
        "| Modelo | Split | AUC | Gini | Log-loss | Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_name, model_metrics in (("CatBoost", cb), ("Logística", lg)):
        for split_name in ("train", "validation", "test"):
            m = model_metrics["splits"][split_name]
            lines.append(f"| {model_name} | {split_name} | {_fmt(m['roc_auc'])} | {_fmt(m['gini'])} | {_fmt(m['log_loss'])} | {_fmt(m['brier_score'])} |")
    lines += [
        "",
        "## 7. Interpretación del Gini",
        "Gini = 2 × AUC − 1. Un Gini alto en train no demuestra por sí solo sobreajuste; debe compararse con validation y test, que representan semanas posteriores no vistas.",
        "",
        "## 8. Diagnóstico de overfitting",
        f"Brecha Gini train-test de CatBoost: {_fmt(overfit_cb)}. Brecha de logística: {_fmt(overfit_lg)}. {diagnosis} La incertidumbre es relevante porque solo hay {ds['positives']} positivos en toda la muestra.",
        "",
        "## 9. Importancia de variables",
        "Se generan importancia global CatBoost, resumen SHAP reducido (sin `case_id`) y coeficientes de la regresión logística. Las explicaciones son agregadas y no contienen probabilidades individuales.",
        "",
        "## 10. Controles",
        f"La probabilidad constante produce Gini 0 por construcción. El target shuffle obtuvo Gini medio de test {_fmt(controls['target_shuffle'].get('mean_test_gini'))}; estado: {controls['target_shuffle'].get('status')}.",
        "",
        "## 11. Limitaciones",
        "La base es una muestra académica, la tasa positiva es baja, las últimas semanas contienen pocos casos, la selección de tablas históricas es deliberadamente ligera y los intervalos bootstrap no sustituyen una validación externa. No se evaluó fairness de forma concluyente ni se usó la base completa.",
        "",
        "## 12. Conclusiones",
        "El proyecto permite estudiar rendimiento, estabilidad temporal, explicabilidad y sobreajuste en una laptop común. Sus resultados sirven para aprendizaje y presentación escolar, no para decisiones crediticias reales ni comparación directa con soluciones ganadoras de Kaggle.",
        "",
        f"**Declaración final:** {DISCLAIMER}",
    ]
    path = output_dir / "ACADEMIC_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

#report.py toma los resultados ya calculados y crea el informe academico final.
#con metricas, interpretacion, advertencias de sobreajuste, controles, limitaciones y conclusiones.