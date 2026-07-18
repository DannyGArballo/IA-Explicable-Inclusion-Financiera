from school_credit.config import PROFILES


def test_smoke_profile_is_light():
    p = PROFILES["smoke"]
    assert p.max_cases == 1000
    assert p.iterations == 100
    assert p.depth == 4
    assert p.shap_rows == 50


def test_school_profile_matches_assignment():
    p = PROFILES["school"]
    assert p.max_cases == 10000
    assert p.iterations == 300
    assert p.depth == 6
    assert p.shap_rows == 200
    assert p.early_stopping_rounds in range(40, 51)
