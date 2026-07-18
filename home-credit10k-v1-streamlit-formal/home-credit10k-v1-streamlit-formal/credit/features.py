#este es archivo encargado de construir, transformar y seleccionar las variables que usaran CatBoost y la regresion logistica. 
#o sea convierte las muchas tablas originales de Home Credit en una sola matriz

from __future__ import annotations
from pathlib import Path
from typing import Any
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from .config import DATE_MARKERS, DIRECT_ID_MARKERS, HISTORICAL_FAMILIES, STRUCTURAL_COLUMNS
from .data import resolve_train_dir

SENSITIVE_TEXT_MARKERS = ("name", "employer", "zipcode", "zip_", "address", "district")


def _is_temporal_column(name: str) -> bool:
    low = name.lower()
    # Home Credit temporal fields use a numeric token followed by D or T.
    suffix_temporal = bool(re.search(r"_[0-9]+[dt]$", low))
    explicit_temporal = any(marker in low for marker in DATE_MARKERS)
    return suffix_temporal or explicit_temporal


def _is_direct_id(name: str) -> bool:
    low = name.lower()
    return name in STRUCTURAL_COLUMNS or any(marker in low for marker in DIRECT_ID_MARKERS)


def _safe_read(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _prefix_columns(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    rename = {c: f"{family}__{c}" for c in frame.columns if c != "case_id"}
    return frame.rename(columns=rename)


def _read_static_family(train_dir: Path, pattern: str, family: str, case_ids: set[int]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    files = sorted(train_dir.glob(pattern))
    if not files:
        return pd.DataFrame({"case_id": list(case_ids)}), {}
    pieces = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in header if c == "case_id" or (not _is_temporal_column(c) and not _is_direct_id(c))]
        part = _safe_read(path, usecols=usecols)
        part = part[part["case_id"].isin(case_ids)]
        pieces.append(part)
    data = pd.concat(pieces, ignore_index=True)
    data = data.drop_duplicates("case_id", keep="first")
    data = _prefix_columns(data, family)
    meta = {}
    for col in data.columns:
        if col == "case_id":
            continue
        source = col.split("__", 1)[1]
        meta[col] = {"source_table": family, "source_column": source, "transformation": "raw depth-0 value"}
    return data, meta


def _candidate_columns(files: list[Path], max_numeric: int = 6, max_categorical: int = 2) -> tuple[list[str], list[str], list[str]]:
    header = pd.read_csv(files[0], nrows=0).columns.tolist()
    structural = [c for c in ("case_id", "num_group1", "num_group2") if c in header]
    sample = pd.read_csv(files[0], nrows=300, low_memory=False)
    numeric: list[str] = []
    categorical: list[str] = []
    for col in header:
        if col in structural or _is_temporal_column(col) or _is_direct_id(col):
            continue
        low = col.lower()
        if any(marker in low for marker in SENSITIVE_TEXT_MARKERS):
            continue
        if col in sample.columns and pd.api.types.is_numeric_dtype(sample[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return structural, numeric[:max_numeric], categorical[:max_categorical]


def _aggregate_historical(train_dir: Path, pattern: str, family: str, case_ids: set[int]) -> tuple[pd.DataFrame | None, dict[str, dict[str, Any]]]:
    files = sorted(train_dir.glob(pattern))
    if not files:
        return None, {}
    structural, numeric, categorical = _candidate_columns(files)
    usecols = list(dict.fromkeys(structural + numeric + categorical))
    pieces = []
    for path in files:
        available = pd.read_csv(path, nrows=0).columns.tolist()
        cols = [c for c in usecols if c in available]
        part = _safe_read(path, usecols=cols)
        part = part[part["case_id"].isin(case_ids)]
        pieces.append(part)
    if not pieces:
        return None, {}
    data = pd.concat(pieces, ignore_index=True)
    sort_cols = [c for c in ("case_id", "num_group1", "num_group2") if c in data.columns]
    data = data.sort_values(sort_cols, kind="stable")
    group = data.groupby("case_id", sort=False, observed=True)
    out = pd.DataFrame(index=group.size().index)
    out[f"{family}__row_count"] = group.size().astype(float)
    meta: dict[str, dict[str, Any]] = {
        f"{family}__row_count": {"source_table": family, "source_column": "*", "transformation": "count"}
    }
    for col in numeric:
        if col not in data.columns:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        temp = data.assign(_value=values).groupby("case_id", sort=False)["_value"]
        aggs = {
            "min": temp.min(), "max": temp.max(), "mean": temp.mean(), "std": temp.std(),
            "first": temp.first(), "last": temp.last(),
        }
        aggs["range"] = aggs["max"] - aggs["min"]
        for agg_name, series in aggs.items():
            feature = f"{family}__{col}__{agg_name}"
            out[feature] = series
            meta[feature] = {"source_table": family, "source_column": col, "transformation": agg_name}
    for col in categorical:
        if col not in data.columns:
            continue
        temp = data.assign(_value=data[col].astype("string")).groupby("case_id", sort=False)["_value"]
        aggs = {"first": temp.first(), "last": temp.last(), "nunique": temp.nunique(dropna=True)}
        for agg_name, series in aggs.items():
            feature = f"{family}__{col}__{agg_name}"
            out[feature] = series
            meta[feature] = {"source_table": family, "source_column": col, "transformation": agg_name}
    return out.reset_index(), meta


def build_feature_matrix(data_dir: str | Path, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict]:
    train_dir = resolve_train_dir(data_dir)
    case_ids = set(base["case_id"].astype(int).tolist())
    matrix = base[[c for c in ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"] if c in base.columns]].copy()
    metadata: dict[str, dict[str, Any]] = {}
    used_tables: list[str] = []

    for pattern, family in (("train_static_0_*.csv", "static_0"), ("train_static_cb_0.csv", "static_cb_0")):
        static, meta = _read_static_family(train_dir, pattern, family, case_ids)
        if len(static.columns) > 1:
            matrix = matrix.merge(static, on="case_id", how="left", validate="one_to_one")
            metadata.update(meta)
            used_tables.append(family)

    for family, pattern in HISTORICAL_FAMILIES.items():
        agg, meta = _aggregate_historical(train_dir, pattern, family, case_ids)
        if agg is not None and len(agg.columns) > 1:
            matrix = matrix.merge(agg, on="case_id", how="left", validate="one_to_one")
            metadata.update(meta)
            used_tables.append(family)

    diagnostics = {
        "rows": int(len(matrix)),
        "columns_total": int(matrix.shape[1]),
        "engineered_features": int(matrix.shape[1] - len([c for c in matrix.columns if c in STRUCTURAL_COLUMNS])),
        "tables_used": used_tables,
    }
    return matrix, metadata, diagnostics


def _score_feature(series: pd.Series, target: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        if x.notna().sum() < 3 or x.dropna().nunique() <= 1:
            return 0.0
        corr = x.corr(target, method="spearman")
        return 0.0 if pd.isna(corr) else abs(float(corr))
    x = series.astype("string").fillna("__MISSING__")
    codes, _ = pd.factorize(x, sort=True)
    return float(mutual_info_score(target, codes))


def select_features_train_only(
    train: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    max_features: int,
    null_threshold: float = 0.95,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    y = train["target"].astype(int)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for col in train.columns:
        if col in STRUCTURAL_COLUMNS or _is_direct_id(col) or _is_temporal_column(col):
            excluded.append({"feature": col, "reason": "structural/direct identifier/temporal"})
            continue
        s = train[col]
        null_pct = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        unique_ratio = nunique / max(1, int(s.notna().sum()))
        reason = None
        if null_pct >= null_threshold:
            reason = f"excluded: null fraction >= {null_threshold:.0%}"
        elif nunique <= 1:
            reason = "excluded: constant on train"
        elif (not pd.api.types.is_numeric_dtype(s)) and nunique > 50 and unique_ratio > 0.98:
            reason = "excluded: nearly unique categorical on train"
        if reason:
            excluded.append({"feature": col, "reason": reason})
            continue
        score = _score_feature(s, y)
        info = metadata.get(col, {"source_table": "unknown", "source_column": col, "transformation": "unknown"})
        candidates.append({
            "feature": col,
            "source_table": info["source_table"],
            "source_column": info["source_column"],
            "transformation": info["transformation"],
            "null_percentage_train": null_pct * 100.0,
            "selection_score_train_only": score,
        })
    ranked = sorted(candidates, key=lambda row: (-row["selection_score_train_only"], row["feature"]))
    selected_rows = ranked[:max_features]
    for rank, row in enumerate(selected_rows, start=1):
        row["inclusion_reason"] = f"retained by train-only relevance ranking (rank {rank}/{len(ranked)})"
    selected = [row["feature"] for row in selected_rows]
    return selected, pd.DataFrame(selected_rows), pd.DataFrame(excluded)


#features.py toma las distintas tablas de Home Credit, excluye fechas e identificadores, resume los historiales mediante conteos y estadisticas.
#une todo en una fila por solicitud y selecciona las variables mas relevantes utilizando exclusivamente los datos de entrenamiento.