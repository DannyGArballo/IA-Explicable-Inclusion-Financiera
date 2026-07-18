#data.py se encarga de localizar, cargar, validar y resumir la tabla principal del dataset, llamada train_base.csv.

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_BASE_COLUMNS = {"case_id", "WEEK_NUM", "target"}


def resolve_train_dir(data_dir: str | Path) -> Path:
    p = Path(data_dir).expanduser().resolve()
    candidates = [p, p / "train", p / "csv_files" / "train"]
    for candidate in candidates:
        if (candidate / "train_base.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"No se encontró train_base.csv. Use la ruta a csv_files o csv_files/train: {p}"
    )


def load_base(data_dir: str | Path) -> pd.DataFrame:
    train_dir = resolve_train_dir(data_dir)
    base = pd.read_csv(train_dir / "train_base.csv", low_memory=False)
    missing = REQUIRED_BASE_COLUMNS - set(base.columns)
    if missing:
        raise ValueError(f"train_base.csv no contiene columnas requeridas: {sorted(missing)}")
    if base["case_id"].duplicated().any():
        raise ValueError("case_id debe ser único en train_base.csv")
    base["target"] = pd.to_numeric(base["target"], errors="raise").astype(int)
    base["WEEK_NUM"] = pd.to_numeric(base["WEEK_NUM"], errors="raise").astype(int)
    if set(base["target"].unique()) - {0, 1}:
        raise ValueError("target debe ser binario 0/1")
    return base.sort_values(["WEEK_NUM", "case_id"]).reset_index(drop=True)


def sample_base_for_profile(base: pd.DataFrame, max_cases: int | None, random_state: int) -> pd.DataFrame:
    if max_cases is None or len(base) <= max_cases:
        return base.copy()
    # Smoke mantiene aproximadamente la prevalencia original; solo reduce volumen.
    ratio = max_cases / len(base)
    n_positive = max(2, int(round(int(base.target.sum()) * ratio)))
    n_positive = min(n_positive, int(base.target.sum()), max_cases - 2)
    n_negative = max_cases - n_positive
    positives = base[base.target == 1].sample(n=n_positive, random_state=random_state)
    negatives = base[base.target == 0].sample(n=n_negative, random_state=random_state)
    picked = pd.concat([positives, negatives], ignore_index=False)
    return picked.sort_values(["WEEK_NUM", "case_id"]).reset_index(drop=True)


def dataset_summary(base: pd.DataFrame) -> dict:
    return {
        "rows": int(len(base)),
        "positives": int(base["target"].sum()),
        "negatives": int((base["target"] == 0).sum()),
        "positive_rate": float(base["target"].mean()),
        "weeks": int(base["WEEK_NUM"].nunique()),
        "week_min": int(base["WEEK_NUM"].min()),
        "week_max": int(base["WEEK_NUM"].max()),
        "case_id_unique": bool(base["case_id"].is_unique),
    }

#data.py prepara la tabla principal: encuentra su ubicacion, valida que tenga las columnas y valores correctos y 
#crea una muestra equilibrada para el perfil rapido (el smoke) y genera el resumen general del dataset.