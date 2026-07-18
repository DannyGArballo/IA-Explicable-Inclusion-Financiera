from school_credit.metrics import gini_from_auc


def test_gini_formula():
    assert gini_from_auc(0.5) == 0.0
    assert gini_from_auc(1.0) == 1.0
    assert gini_from_auc(0.75) == 0.5
