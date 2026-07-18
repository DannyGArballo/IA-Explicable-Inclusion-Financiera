from pathlib import Path
import pandas as pd
import pytest
from school_credit.data import load_base, resolve_train_dir


def test_load_home_credit_layout(tmp_path: Path):
    train = tmp_path / "csv_files" / "train"
    train.mkdir(parents=True)
    pd.DataFrame({"case_id": [1, 2], "WEEK_NUM": [0, 1], "target": [0, 1]}).to_csv(train / "train_base.csv", index=False)
    assert resolve_train_dir(tmp_path / "csv_files") == train.resolve()
    loaded = load_base(tmp_path / "csv_files")
    assert len(loaded) == 2
    assert loaded.target.sum() == 1


def test_duplicate_case_id_rejected(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    pd.DataFrame({"case_id": [1, 1], "WEEK_NUM": [0, 1], "target": [0, 1]}).to_csv(train / "train_base.csv", index=False)
    with pytest.raises(ValueError, match="único"):
        load_base(train)
