from pathlib import Path

import numpy as np

from streamlit_app.inference import (
    classify_decision,
    load_resources,
    predict_with_explanation,
    scenario_values,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "streamlit_app" / "artifacts"


def test_serialized_model_predicts_and_explains():
    model, metadata = load_resources(ARTIFACTS)
    values = scenario_values(metadata, "Perfil típico")
    score, explanation = predict_with_explanation(model, metadata, values)
    assert 0.0 <= score <= 1.0
    assert len(explanation) == len(metadata["selected_features"])
    assert np.isfinite(explanation["contribucion_shap"]).all()


def test_scenarios_produce_valid_scores():
    model, metadata = load_resources(ARTIFACTS)
    scores = []
    for scenario in ["Perfil típico", "Ejemplo de menor puntaje", "Ejemplo de mayor puntaje"]:
        values = scenario_values(metadata, scenario)
        score, _ = predict_with_explanation(model, metadata, values)
        assert 0.0 <= score <= 1.0
        scores.append(score)
    assert len(set(round(score, 8) for score in scores)) >= 2


def test_classify_decision_has_three_zones():
    assert classify_decision(0.20, 0.40)[1] == "success"
    assert classify_decision(0.40, 0.40)[1] == "warning"
    assert classify_decision(0.60, 0.40)[1] == "error"
