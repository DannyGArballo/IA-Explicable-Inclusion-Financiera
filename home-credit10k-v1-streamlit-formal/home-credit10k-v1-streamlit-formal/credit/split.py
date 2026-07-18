from __future__ import annotations
import numpy as np
import pandas as pd


def temporal_split(frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> tuple[dict[str, pd.DataFrame], dict]:
    weeks = np.array(sorted(frame["WEEK_NUM"].unique()))
    if len(weeks) < 3:
        raise ValueError("Se requieren al menos tres semanas para un split temporal")
    grouped = frame.groupby("WEEK_NUM")["target"].agg(["size", "sum"]).reindex(weeks)
    cum_rows = grouped["size"].cumsum().to_numpy()
    total = len(frame)
    target1 = total * train_fraction
    target2 = total * (train_fraction + validation_fraction)
    best = None
    min_positive = 2
    for i in range(0, len(weeks) - 2):
        for j in range(i + 1, len(weeks) - 1):
            masks = [frame.WEEK_NUM <= weeks[i], (frame.WEEK_NUM > weeks[i]) & (frame.WEEK_NUM <= weeks[j]), frame.WEEK_NUM > weeks[j]]
            stats = [(int(m.sum()), int(frame.loc[m, "target"].sum())) for m in masks]
            valid = all(n > 0 and 0 < p < n and p >= min_positive for n, p in stats)
            penalty = 0 if valid else 1000
            objective = abs(cum_rows[i] - target1) / total + abs(cum_rows[j] - target2) / total + penalty
            if best is None or objective < best[0]:
                best = (objective, weeks[i], weeks[j], stats, valid)
    assert best is not None
    _, train_end, val_end, _, valid = best
    if not valid:
        raise ValueError("No fue posible construir splits temporales con ambas clases")
    splits = {
        "train": frame[frame.WEEK_NUM <= train_end].copy(),
        "validation": frame[(frame.WEEK_NUM > train_end) & (frame.WEEK_NUM <= val_end)].copy(),
        "test": frame[frame.WEEK_NUM > val_end].copy(),
    }
    ids = {k: set(v.case_id.tolist()) for k, v in splits.items()}
    overlap = bool(ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    if overlap:
        raise ValueError("Se detectó solapamiento de case_id entre splits")
    summary_rows = []
    for name, part in splits.items():
        summary_rows.append({
            "split": name,
            "rows": int(len(part)),
            "positives": int(part.target.sum()),
            "negatives": int((part.target == 0).sum()),
            "positive_rate": float(part.target.mean()),
            "week_min": int(part.WEEK_NUM.min()),
            "week_max": int(part.WEEK_NUM.max()),
            "weeks": int(part.WEEK_NUM.nunique()),
            "both_classes": bool(part.target.nunique() == 2),
        })
    info = {"train_end_week": int(train_end), "validation_end_week": int(val_end), "case_id_overlap": overlap, "splits": summary_rows}
    return splits, info

#split.py divide cronologicamente las solicitudes. Busca puntos de corte cercanos a una proporcion 70/15/15, mantiene cada semana completa 
#exige positivos y negativos en las tres partes y verifica que ninguna solicitud se repita. Esto permite entrenar con el pasado y evaluar con periodos posteriores.
