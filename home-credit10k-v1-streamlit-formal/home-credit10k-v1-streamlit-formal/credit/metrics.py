from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def gini_from_auc(auc: float) -> float:
    return 2.0 * float(auc) - 1.0


def _bootstrap(y: np.ndarray, p: np.ndarray, repeats: int, random_state: int) -> dict | None:
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if len(positives) < 2 or len(negatives) < 2:
        return None
    rng = np.random.default_rng(random_state)
    aucs = []
    for _ in range(repeats):
        idx = np.concatenate([
            rng.choice(positives, size=len(positives), replace=True),
            rng.choice(negatives, size=len(negatives), replace=True),
        ])
        aucs.append(float(roc_auc_score(y[idx], p[idx])))
    arr = np.asarray(aucs)
    return {
        "auc_lower_2_5": float(np.percentile(arr, 2.5)),
        "auc_median": float(np.median(arr)),
        "auc_upper_97_5": float(np.percentile(arr, 97.5)),
        "gini_lower_2_5": gini_from_auc(float(np.percentile(arr, 2.5))),
        "gini_median": gini_from_auc(float(np.median(arr))),
        "gini_upper_97_5": gini_from_auc(float(np.percentile(arr, 97.5))),
        "repeats": int(repeats),
    }


def weekly_metrics(y: np.ndarray, p: np.ndarray, weeks: np.ndarray) -> dict:
    rows = []
    for week in sorted(np.unique(weeks)):
        mask = weeks == week
        yw = y[mask]
        if len(np.unique(yw)) < 2:
            continue
        auc = float(roc_auc_score(yw, p[mask]))
        rows.append({"week": int(week), "rows": int(mask.sum()), "positives": int(yw.sum()), "auc": auc, "gini": gini_from_auc(auc)})
    if not rows:
        return {"weekly": [], "valid_weeks": 0, "mean_gini": None, "std_gini": None, "trend": None, "stability_score": None}
    ginis = np.array([row["gini"] for row in rows])
    x = np.arange(len(ginis), dtype=float)
    trend = float(np.polyfit(x, ginis, 1)[0]) if len(ginis) >= 2 else None
    std = float(ginis.std(ddof=0))
    mean = float(ginis.mean())
    stability = float(mean - 0.5 * std - (abs(trend) if trend is not None else 0.0))
    return {"weekly": rows, "valid_weeks": len(rows), "mean_gini": mean, "std_gini": std, "trend": trend, "stability_score": stability}


def evaluate_split(y: pd.Series, p: np.ndarray, weeks: pd.Series, split: str, bootstrap_repeats: int, random_state: int) -> dict:
    yt = y.to_numpy(dtype=int)
    pred = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    auc = float(roc_auc_score(yt, pred)) if len(np.unique(yt)) == 2 else None
    result = {
        "split": split,
        "rows": int(len(yt)),
        "positives": int(yt.sum()),
        "negatives": int((yt == 0).sum()),
        "roc_auc": auc,
        "gini": gini_from_auc(auc) if auc is not None else None,
        "log_loss": float(log_loss(yt, pred, labels=[0, 1])),
        "brier_score": float(brier_score_loss(yt, pred)),
        "bootstrap": _bootstrap(yt, pred, bootstrap_repeats, random_state) if split in {"validation", "test"} else None,
    }
    result.update(weekly_metrics(yt, pred, weeks.to_numpy()))
    return result


def evaluate_model(splits: dict[str, pd.DataFrame], predictions: dict[str, np.ndarray], bootstrap_repeats: int, random_state: int) -> dict:
    by_split = {
        name: evaluate_split(part.target, predictions[name], part.WEEK_NUM, name, bootstrap_repeats, random_state)
        for name, part in splits.items()
    }
    train = by_split["train"]
    val = by_split["validation"]
    test = by_split["test"]
    gaps = {
        "train_minus_validation_gini": float(train["gini"] - val["gini"]),
        "train_minus_test_gini": float(train["gini"] - test["gini"]),
        "train_minus_validation_auc": float(train["roc_auc"] - val["roc_auc"]),
        "train_minus_test_auc": float(train["roc_auc"] - test["roc_auc"]),
    }
    return {"splits": by_split, "gaps": gaps}

#metrics.py calcula el rendimiento de CatBoost y de la regresion logistica. 
#a partir de las etiquetas reales y los puntajes predichos, produce la informacion que luego aparece en metrics.json.