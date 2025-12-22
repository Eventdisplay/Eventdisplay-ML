
import eventdisplay_ml.training_variables


def test_xgb_per_telescope_training_variables():
    vars = eventdisplay_ml.training_variables.xgb_per_telescope_training_variables()
    assert isinstance(vars, list)
    assert "Disp_T" in vars
    assert "R_core" in vars


def test_xgb_array_training_variables():
    vars = eventdisplay_ml.training_variables.xgb_array_training_variables()
    assert isinstance(vars, list)
    assert "DispNImages" in vars
    assert "EmissionHeight" in vars


def test_xgb_all_training_variables():
    vars = eventdisplay_ml.training_variables.xgb_all_training_variables()
    assert isinstance(vars, list)
    assert "Disp_T" in vars
    assert "R_core" in vars
    assert "DispNImages" in vars
    assert "EmissionHeight" in vars