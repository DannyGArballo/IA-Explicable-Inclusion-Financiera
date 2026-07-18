import numpy as np
import pandas as pd
from school_credit.features import select_features_train_only


def base_train():
    rng = np.random.default_rng(42)
    n = 200
    target = np.tile([0, 1], n // 2)
    return pd.DataFrame({
        "case_id": np.arange(n),
        "WEEK_NUM": np.repeat(np.arange(20), 10),
        "target": target,
        "good": target + rng.normal(0, 0.2, n),
        "noise": rng.normal(size=n),
        "constant": 1,
        "almost_null": [np.nan] * 195 + [1] * 5,
        "person_id": np.arange(n),
        "date_createdD": "2020-01-01",
        "category": np.where(target == 1, "A", "B"),
    })


def test_selection_uses_train_input_only_and_removes_ids():
    train = base_train()
    selected, selected_df, excluded = select_features_train_only(train, {}, max_features=3)
    assert "case_id" not in selected
    assert "person_id" not in selected
    assert "date_createdD" not in selected
    assert "constant" not in selected
    assert "almost_null" not in selected
    assert "good" in selected
    assert len(selected) <= 3
    assert len(selected_df) == len(selected)
    assert not excluded.empty


def test_feature_limit_is_enforced():
    train = base_train()
    for i in range(20):
        train[f"f{i}"] = np.arange(len(train)) % (i + 2)
    selected, _, _ = select_features_train_only(train, {}, max_features=5)
    assert len(selected) == 5
