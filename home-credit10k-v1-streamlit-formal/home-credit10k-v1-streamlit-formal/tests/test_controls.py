import numpy as np
import pandas as pd
from school_credit.controls import target_shuffle_control


def test_target_shuffle_control_runs():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=300), "target": np.tile([0, 0, 0, 1], 75), "WEEK_NUM": 0})
    test = pd.DataFrame({"x": rng.normal(size=120), "target": np.tile([0, 0, 0, 1], 30), "WEEK_NUM": 1})
    splits = {"train": train, "validation": test.copy(), "test": test}
    result = target_shuffle_control(splits, ["x"], repeats=1, random_state=42)
    assert result["status"] in {"ok", "warning"}
    assert np.isfinite(result["mean_test_gini"])
