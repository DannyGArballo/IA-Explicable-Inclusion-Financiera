from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .metrics import gini_from_auc

#Su funcion es comprobar que los resultados del proyecto tengan sentido y que el pipeline no este encontrando capacidad predictiva donde no deberia de existir.

def constant_probability_control(splits: dict[str, pd.DataFrame]) -> dict:
    prevalence = float(splits["train"].target.mean())
    output = {"train_prevalence": prevalence, "splits": {}}
    for name, part in splits.items():
        pred = np.full(len(part), prevalence)
        auc = float(roc_auc_score(part.target, pred))
        output["splits"][name] = {"auc": auc, "gini": gini_from_auc(auc)}
    return output


def target_shuffle_control(splits: dict[str, pd.DataFrame], features: list[str], repeats: int, random_state: int) -> dict:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(splits["train"][c])]
    numeric = numeric[:100]
    if not numeric:
        return {"status": "not_available", "reason": "no numeric features"}
    rng = np.random.default_rng(random_state)
    ginis = []
    for _ in range(repeats):
        shuffled = rng.permutation(splits["train"].target.to_numpy())
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.2, solver="liblinear", class_weight="balanced", max_iter=300, random_state=random_state)),
        ])
        pipe.fit(splits["train"][numeric], shuffled)
        pred = pipe.predict_proba(splits["test"][numeric])[:, 1]
        ginis.append(gini_from_auc(float(roc_auc_score(splits["test"].target, pred))))
    mean_gini = float(np.mean(ginis))
    warning = abs(mean_gini) > 0.20
    return {"status": "warning" if warning else "ok", "test_ginis": ginis, "mean_test_gini": mean_gini, "warning": warning, "threshold_abs_gini": 0.20}

#controls.py tiene ruebas negativas de sanidad. Compara los modelos contra una prediccion constante y verifica que al revolver las varables o etiquetas
#desaparezca la señal predictiva. 
#Esto ayuda a detectar resultados sospechosos, errores o posibles fugas de informacion.