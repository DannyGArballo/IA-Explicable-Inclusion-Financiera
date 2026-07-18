from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class CatBoostResult:
    models: list[CatBoostClassifier]
    predictions: dict[str, np.ndarray]
    categorical_columns: list[str]
    best_iterations: list[int]

@dataclass
class LogisticResult:
    pipeline: Pipeline
    predictions: dict[str, np.ndarray]
    coefficients: pd.DataFrame


def prepare_catboost_frames(splits: dict[str, pd.DataFrame], features: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    train_x = splits["train"][features]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(train_x[c])]
    prepared: dict[str, pd.DataFrame] = {}
    for name, part in splits.items():
        x = part[features].copy()
        for col in categorical:
            x[col] = x[col].astype("string").fillna("__MISSING__").astype(str)
        for col in set(features) - set(categorical):
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
        prepared[name] = x
    return prepared, categorical


def train_catboost(
    splits: dict[str, pd.DataFrame], features: list[str], iterations: int, depth: int,
    early_stopping_rounds: int, seeds: tuple[int, ...], threads: int,
) -> CatBoostResult:
    frames, categorical = prepare_catboost_frames(splits, features)
    cat_indices = [features.index(c) for c in categorical]
    pools = {
        name: Pool(frames[name], label=splits[name]["target"].to_numpy(), cat_features=cat_indices, feature_names=features)
        for name in splits
    }
    models: list[CatBoostClassifier] = []
    per_seed: list[dict[str, np.ndarray]] = []
    best_iterations: list[int] = []
    for seed in seeds:
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=max(1, threads),
            l2_leaf_reg=5.0,
            random_strength=0.5,
            auto_class_weights="Balanced",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(pools["train"], eval_set=pools["validation"], early_stopping_rounds=early_stopping_rounds, use_best_model=True)
        models.append(model)
        best_iterations.append(int(model.get_best_iteration()))
        per_seed.append({name: model.predict_proba(pool)[:, 1] for name, pool in pools.items()})
    predictions = {name: np.mean([pred[name] for pred in per_seed], axis=0) for name in splits}
    return CatBoostResult(models, predictions, categorical, best_iterations)


def train_logistic(splits: dict[str, pd.DataFrame], features: list[str], random_state: int) -> LogisticResult:
    train_x = splits["train"][features]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train_x[c])]
    categorical = [c for c in features if c not in numeric]
    clean_frames: dict[str, pd.DataFrame] = {}
    for name, part in splits.items():
        x = part[features].copy()
        for col in numeric:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        for col in categorical:
            # sklearn SimpleImputer expects np.nan rather than pandas.NA for object arrays.
            x[col] = x[col].astype(object).where(pd.notna(x[col]), np.nan)
        clean_frames[name] = x
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)),
        ]), categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    model = LogisticRegression(
        C=0.5, solver="liblinear", class_weight="balanced",
        max_iter=600, random_state=random_state,
    )
    pipeline = Pipeline([("preprocess", pre), ("model", model)])
    pipeline.fit(clean_frames["train"], splits["train"]["target"])
    predictions = {name: pipeline.predict_proba(clean_frames[name])[:, 1] for name in splits}
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    coef = pd.DataFrame({"encoded_feature": names, "coefficient": coefs})
    coef["absolute_magnitude"] = coef.coefficient.abs()
    coef["sign"] = np.where(coef.coefficient > 0, "positive", np.where(coef.coefficient < 0, "negative", "zero"))
    coef["near_zero"] = coef.absolute_magnitude < 1e-6
    coef = coef.sort_values("absolute_magnitude", ascending=False).reset_index(drop=True)
    return LogisticResult(pipeline, predictions, coef)

#models.py es el archivo encargado de preparar los datos, entrenar CatBoost y la regresion logistica, y generar sus predicciones para entrenamiento, validacion y prueba.
