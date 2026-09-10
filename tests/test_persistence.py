import pytest
from fragrance_ai.recommender.persistence import model_persistence


def curve(*pairs):
    return [{"minutes": time, "relative_to_opening_intensity_percent": value} for time, value in pairs]


@pytest.mark.parametrize("required,expected", [(50, True), (90, None), (130, False)])
def test_crossing_reports_estimate_and_conservative_sample_bracket(required, expected):
    result = model_persistence(curve((0, 100), (60, 70), (120, 20)), 50, required)
    assert result["estimate_minutes"] == 84
    assert result["lower_bound_minutes"] == 60 and result["upper_bound_minutes"] == 120
    assert result["meets_requirement"] is expected
    assert not result["human_detection_guaranteed"]


def test_right_censoring_does_not_invent_an_exact_lifetime():
    points = curve((0, 100), (480, 20))
    result = model_persistence(points, 10, 480)
    assert result["status"] == "right_censored"
    assert result["estimate_minutes"] is None and result["upper_bound_minutes"] is None
    assert result["lower_bound_minutes"] == 480 and result["meets_requirement"] is True
    assert model_persistence(points, 10, 600)["meets_requirement"] is None


def test_reappearance_does_not_erase_the_first_loss():
    result = model_persistence(curve((0, 100), (60, 0), (120, 100)), 50, 90)
    assert result["estimate_minutes"] == 30 and result["later_reappearance"]
    assert result["meets_requirement"] is False


def test_initially_below_and_exactly_on_threshold_are_distinguished():
    assert model_persistence(curve((0, 5), (60, 20)), 10, 1)["status"] == "below_threshold_at_start"
    assert model_persistence(curve((0, 10), (60, 0)), 10, 1)["status"] == "crossing_estimated"


@pytest.mark.parametrize("points", [[], curve((0, 100)), curve((1, 100), (60, 50)),
    curve((0, 100), (0, 50)), curve((0, 100), (60, float("nan")))])
def test_missing_or_invalid_timepoints_never_pass(points):
    result = model_persistence(points, 10, 60)
    assert result["status"] == "unavailable" and result["meets_requirement"] is None


@pytest.mark.parametrize("threshold", [True, 0, -1, 101, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        model_persistence(curve((0, 100), (60, 10)), threshold)
