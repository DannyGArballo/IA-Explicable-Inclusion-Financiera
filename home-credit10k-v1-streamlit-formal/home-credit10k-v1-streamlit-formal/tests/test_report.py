from pathlib import Path
from school_credit.report import DISCLAIMER, write_report


def model_metrics():
    split = {"roc_auc": 0.7, "gini": 0.4, "log_loss": 0.2, "brier_score": 0.1}
    return {"splits": {"train": split, "validation": split, "test": split}, "gaps": {"train_minus_test_gini": 0.0}}


def test_report_generation(tmp_path: Path):
    payload = {
        "dataset": {"rows": 10000, "positives": 314, "weeks": 92},
        "split": {"train_end_week": 51, "validation_end_week": 67},
        "metrics": {"catboost": model_metrics(), "logistic": model_metrics()},
        "controls": {"target_shuffle": {"mean_test_gini": 0.01, "status": "ok"}},
    }
    path = write_report(tmp_path, payload)
    text = path.read_text(encoding="utf-8")
    assert DISCLAIMER in text
    assert "sobreajuste" in text
