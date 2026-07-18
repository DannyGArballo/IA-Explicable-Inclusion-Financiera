from __future__ import annotations
from pathlib import Path
import platform
import sys
import time
import logging
import importlib.metadata
import pandas as pd
import psutil

from .config import PROFILES
from .controls import constant_probability_control, target_shuffle_control
from .data import dataset_summary, load_base, sample_base_for_profile
from .explain import generate_catboost_explanations, generate_logistic_explanations
from .features import build_feature_matrix, select_features_train_only
from .metrics import evaluate_model
from .models import train_catboost, train_logistic
from .report import DISCLAIMER, write_json, write_report
from .split import temporal_split


def _versions() -> dict:
    names = ["pandas", "numpy", "scikit-learn", "catboost", "matplotlib", "psutil", "pytest"]
    return {name: importlib.metadata.version(name) for name in names}


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _flat_metrics(model: str, metrics: dict) -> list[dict]:
    rows = []
    for split, m in metrics["splits"].items():
        rows.append({
            "model": model, "split": split, "rows": m["rows"], "positives": m["positives"],
            "roc_auc": m["roc_auc"], "gini": m["gini"], "log_loss": m["log_loss"],
            "brier_score": m["brier_score"], "mean_weekly_gini": m["mean_gini"],
            "std_weekly_gini": m["std_gini"], "gini_trend": m["trend"],
            "stability_score": m["stability_score"],
        })
    return rows


def run_pipeline(data_dir: str | Path, profile_name: str, output_dir: str | Path, random_state: int = 42, max_features: int = 250, threads: int = 2) -> dict:
    if profile_name not in PROFILES:
        raise ValueError(f"Perfil inválido: {profile_name}")
    profile = PROFILES[profile_name]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("school_pipeline")
    handler = logging.FileHandler(output / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    timings: dict[str, dict] = {}
    start_all = time.perf_counter()

    def stage(name: str, started: float) -> None:
        timings[name] = {"seconds": time.perf_counter() - started, "memory_rss_mb": _memory_mb()}
        logger.info("%s completed in %.2fs | RSS %.1f MB", name, timings[name]["seconds"], timings[name]["memory_rss_mb"])

    t = time.perf_counter()
    full_base = load_base(data_dir)
    full_summary = dataset_summary(full_base)
    base = sample_base_for_profile(full_base, profile.max_cases, random_state)
    run_summary = dataset_summary(base)
    stage("load_and_profile_base", t)

    t = time.perf_counter()
    matrix, lineage, feature_diagnostics = build_feature_matrix(data_dir, base)
    stage("feature_engineering", t)

    t = time.perf_counter()
    splits, split_info = temporal_split(matrix)
    split_df = pd.DataFrame(split_info["splits"])
    split_df.to_csv(output / "split_summary.csv", index=False)
    stage("temporal_split", t)

    t = time.perf_counter()
    selected, selected_df, excluded_df = select_features_train_only(splits["train"], lineage, max_features=max_features)
    if not selected:
        raise RuntimeError("La selección de variables no retuvo ninguna feature")
    selected_df.to_csv(output / "selected_features.csv", index=False)
    excluded_df.to_csv(output / "excluded_features.csv", index=False)
    stage("train_only_feature_selection", t)

    t = time.perf_counter()
    cb_result = train_catboost(
        splits, selected, profile.iterations, profile.depth, profile.early_stopping_rounds,
        profile.ensemble_seeds, threads,
    )
    cb_metrics = evaluate_model(splits, cb_result.predictions, profile.bootstrap_repeats, random_state)
    stage("catboost_training_and_metrics", t)

    t = time.perf_counter()
    logit_result = train_logistic(splits, selected, random_state)
    logit_metrics = evaluate_model(splits, logit_result.predictions, profile.bootstrap_repeats, random_state)
    stage("logistic_training_and_metrics", t)

    t = time.perf_counter()
    controls = {
        "constant_probability": constant_probability_control(splits),
        "target_shuffle": target_shuffle_control(splits, selected, profile.shuffle_repeats, random_state),
    }
    stage("negative_controls", t)

    t = time.perf_counter()
    explain_dir = output / "explainability"
    cat_exp = generate_catboost_explanations(cb_result, splits, selected, explain_dir, profile.shap_rows, random_state)
    logit_exp = generate_logistic_explanations(logit_result.coefficients, explain_dir)
    stage("explainability", t)

    metrics = {"catboost": cb_metrics, "logistic": logit_metrics}
    metrics_rows = _flat_metrics("catboost", cb_metrics) + _flat_metrics("logistic", logit_metrics)
    pd.DataFrame(metrics_rows).to_csv(output / "metrics.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "controls.json", controls)
    write_json(output / "feature_diagnostics.json", feature_diagnostics)
    write_json(output / "split_info.json", split_info)

    total_seconds = time.perf_counter() - start_all
    timings["total"] = {"seconds": total_seconds, "memory_rss_mb": _memory_mb()}
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _versions(),
        "threads": threads,
    }
    write_json(output / "environment.json", environment)
    write_json(output / "timings.json", timings)

    payload = {
        "disclaimer": DISCLAIMER,
        "profile": profile_name,
        "dataset": run_summary,
        "full_dataset_reference": full_summary,
        "feature_counts": {"initial": feature_diagnostics["engineered_features"], "selected": len(selected)},
        "split": split_info,
        "metrics": metrics,
        "controls": controls,
        "catboost": {"best_iterations": cb_result.best_iterations, "categorical_features": len(cb_result.categorical_columns)},
        "explainability": {"catboost": cat_exp, "logistic": logit_exp},
        "timings": timings,
        "environment": environment,
    }
    write_json(output / "run_summary.json", payload)
    write_report(output, payload)
    logger.info("Pipeline completed in %.2fs", total_seconds)
    return payload

#pipeline.py dirige la ejecucion del proyecto. Carga la base, construye y selecciona variables, realiza el split temporal, entrena los dos modelos, calcula metricas.
#ejecuta controles, genera explicaciones y guarda todos los reportes en una carpeta organizada.