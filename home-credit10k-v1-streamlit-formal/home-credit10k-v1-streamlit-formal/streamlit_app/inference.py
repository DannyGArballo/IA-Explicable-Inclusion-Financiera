from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


class InferenceError(RuntimeError):
    """Error de carga o inferencia de la demostración."""


def load_resources(artifact_dir: str | Path) -> tuple[CatBoostClassifier, dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "catboost_model.cbm"
    metadata_path = artifact_path / "model_metadata.json"
    if not model_path.is_file():
        raise InferenceError(f"No se encontró el modelo: {model_path}")
    if not metadata_path.is_file():
        raise InferenceError(f"No se encontraron los metadatos: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model, metadata


def default_values(metadata: dict[str, Any]) -> dict[str, Any]:
    specs = metadata["feature_specs"]
    return {
        feature: specs[feature].get("default")
        for feature in metadata["selected_features"]
    }


def scenario_values(metadata: dict[str, Any], scenario: str) -> dict[str, Any]:
    values = default_values(metadata)
    if scenario == "Perfil típico":
        return values

    for item in metadata["top_editable_features"]:
        feature = item["feature"]
        spec = metadata["feature_specs"][feature]
        if spec["kind"] != "numeric":
            continue
        direction = item.get("direction", "")
        high_raises = "elevan" in direction or "elevar" in direction
        high_reduces = "reducen" in direction or "reducir" in direction
        if scenario == "Ejemplo de menor puntaje":
            if high_raises:
                values[feature] = spec.get("p25", spec["default"])
            elif high_reduces:
                values[feature] = spec.get("p75", spec["default"])
        elif scenario == "Ejemplo de mayor puntaje":
            if high_raises:
                values[feature] = spec.get("p75", spec["default"])
            elif high_reduces:
                values[feature] = spec.get("p25", spec["default"])
    return values


def prepare_frame(metadata: dict[str, Any], values: dict[str, Any]) -> pd.DataFrame:
    features = metadata["selected_features"]
    categorical = set(metadata["categorical_features"])
    row = {feature: values.get(feature, metadata["feature_specs"][feature].get("default")) for feature in features}
    frame = pd.DataFrame([row], columns=features)

    for feature in features:
        if feature in categorical:
            value = frame.at[0, feature]
            frame.at[0, feature] = "__MISSING__" if pd.isna(value) else str(value)
            frame[feature] = frame[feature].astype(str)
        else:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(float)
    return frame


def predict_with_explanation(
    model: CatBoostClassifier,
    metadata: dict[str, Any],
    values: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    frame = prepare_frame(metadata, values)
    features = metadata["selected_features"]
    categorical = metadata["categorical_features"]
    cat_indices = [features.index(feature) for feature in categorical]
    pool = Pool(frame, cat_features=cat_indices, feature_names=features)

    score = float(model.predict_proba(pool)[0, 1])
    shap_values = model.get_feature_importance(pool, type="ShapValues")[0, :-1]
    specs = metadata["feature_specs"]
    explanation = pd.DataFrame(
        {
            "variable_tecnica": features,
            "variable": [specs[f]["label"] for f in features],
            "contribucion_shap": shap_values,
            "impacto_absoluto": np.abs(shap_values),
        }
    ).sort_values("impacto_absoluto", ascending=False, ignore_index=True)
    return score, explanation


def relative_level(score: float, distribution: dict[str, float]) -> str:
    if score < distribution["p25"]:
        return "Bajo dentro de la muestra"
    if score > distribution["p75"]:
        return "Alto dentro de la muestra"
    return "Intermedio dentro de la muestra"


def classify_decision(score: float, threshold: float, review_band: float = 0.03) -> tuple[str, str]:
    if score <= threshold - review_band:
        return "Aprobación preliminar", "success"
    if score < threshold + review_band:
        return "Revisión adicional", "warning"
    return "No aprobado", "error"
