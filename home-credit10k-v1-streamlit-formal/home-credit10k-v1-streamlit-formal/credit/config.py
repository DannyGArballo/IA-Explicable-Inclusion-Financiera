from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Profile:
    name: str
    max_cases: int | None
    iterations: int
    depth: int
    early_stopping_rounds: int
    shap_rows: int
    bootstrap_repeats: int
    shuffle_repeats: int
    ensemble_seeds: tuple[int, ...]

PROFILES = {
    "smoke": Profile("smoke", 1000, 100, 4, 20, 50, 50, 3, (42,)),
    "school": Profile("school", 10000, 300, 6, 45, 200, 200, 10, (42,)),
    "ensemble": Profile("ensemble", 10000, 300, 6, 45, 200, 200, 10, (42, 43, 44)),
}

STRUCTURAL_COLUMNS = {"case_id", "target", "WEEK_NUM", "MONTH", "date_decision", "num_group1", "num_group2"}
DATE_MARKERS = ("date", "month", "week", "decision")
DIRECT_ID_MARKERS = ("case_id", "num_group", "_id", "identifier")

HISTORICAL_FAMILIES = {
    "applprev_1": "train_applprev_1_*.csv",
    "credit_bureau_a_1": "train_credit_bureau_a_1_*.csv",
    "credit_bureau_a_2": "train_credit_bureau_a_2_*.csv",
    "credit_bureau_b_1": "train_credit_bureau_b_1.csv",
    "person_1": "train_person_1.csv",
    "debitcard_1": "train_debitcard_1.csv",
    "deposit_1": "train_deposit_1.csv",
    "other_1": "train_other_1.csv",
    "tax_registry_a_1": "train_tax_registry_a_1.csv",
    "tax_registry_b_1": "train_tax_registry_b_1.csv",
    "tax_registry_c_1": "train_tax_registry_c_1.csv",
}

#Define como se ejecutan los perfiles smoke, resultados y ensemble, ademas de indicar que columnas y tablas debe reconocer el pipeline.