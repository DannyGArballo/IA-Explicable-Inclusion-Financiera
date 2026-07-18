from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import Pool
from .models import CatBoostResult, prepare_catboost_frames


def generate_catboost_explanations(
    result: CatBoostResult, splits: dict[str, pd.DataFrame], features: list[str], output_dir: Path,
    max_rows: int, random_state: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = result.models[0]
    frames, categorical = prepare_catboost_frames(splits, features)
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance(type="FeatureImportance"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "catboost_feature_importance.csv", index=False)

    sample = frames["test"].sample(min(max_rows, len(frames["test"])), random_state=random_state)
    cat_indices = [features.index(c) for c in categorical]
    pool = Pool(sample, cat_features=cat_indices, feature_names=features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    shap_rank.head(30).to_csv(output_dir / "catboost_shap_top30.csv", index=False)

    top = importance.head(30).iloc[::-1]
    plt.figure(figsize=(9, 8))
    plt.barh(top.feature, top.importance)
    plt.xlabel("CatBoost feature importance")
    plt.tight_layout()
    plt.savefig(output_dir / "catboost_global_importance_top30.png", dpi=150)
    plt.close()

    top_features = shap_rank.head(20).feature.tolist()[::-1]
    indices = [features.index(f) for f in top_features]
    plt.figure(figsize=(10, 8))
    rng = np.random.default_rng(random_state)
    for y_pos, idx in enumerate(indices):
        jitter = rng.normal(0, 0.08, size=shap_values.shape[0])
        plt.scatter(shap_values[:, idx], np.full(shap_values.shape[0], y_pos) + jitter, s=10, alpha=0.55)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("SHAP value (CatBoost native)")
    plt.title(f"Reduced SHAP summary ({len(sample)} observations)")
    plt.tight_layout()
    plt.savefig(output_dir / "catboost_shap_summary.png", dpi=150)
    plt.close()
    return {"shap_rows": int(len(sample)), "top_feature": str(shap_rank.iloc[0].feature), "top30_file": "catboost_shap_top30.csv"}


def generate_logistic_explanations(coefficients: pd.DataFrame, output_dir: Path) -> dict:
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)
    top = coefficients.head(30).iloc[::-1]
    plt.figure(figsize=(9, 8))
    plt.barh(top.encoded_feature, top.coefficient)
    plt.xlabel("Regularized logistic coefficient")
    plt.tight_layout()
    plt.savefig(output_dir / "logistic_coefficients_top30.png", dpi=150)
    plt.close()
    return {"encoded_features": int(len(coefficients)), "near_zero": int(coefficients.near_zero.sum())}

#Toma modelos ya entrenados y crea los archivos CSV y las imagenes a la ruta que le pongas: