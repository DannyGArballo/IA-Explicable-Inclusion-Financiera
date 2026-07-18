import pandas as pd
from school_credit.split import temporal_split


def make_frame():
    rows = []
    case_id = 0
    for week in range(20):
        for i in range(20):
            rows.append({"case_id": case_id, "WEEK_NUM": week, "target": 1 if i == 0 or (week % 5 == 0 and i == 1) else 0, "x": i})
            case_id += 1
    return pd.DataFrame(rows)


def test_temporal_split_has_both_classes_and_no_overlap():
    splits, info = temporal_split(make_frame())
    assert not info["case_id_overlap"]
    assert all(part.target.nunique() == 2 for part in splits.values())
    assert splits["train"].WEEK_NUM.max() < splits["validation"].WEEK_NUM.min()
    assert splits["validation"].WEEK_NUM.max() < splits["test"].WEEK_NUM.min()
    assert not (set(splits["train"].case_id) & set(splits["test"].case_id))
