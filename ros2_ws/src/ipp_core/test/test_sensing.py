from ipp_core.sensing import gamma_sum, SensorModel, UAV_SENSOR, UGV_SENSOR

import numpy as np
import pytest


def test_platform_asymmetry_holds():
    assert UAV_SENSOR.sigma > UGV_SENSOR.sigma
    assert UAV_SENSOR.rho < UGV_SENSOR.rho
    assert UGV_SENSOR.occluded
    assert not UAV_SENSOR.occluded


def test_rate_decays_with_range():
    near = UGV_SENSOR.gamma_at([[1.0, 6.0]], (0.0, 6.0))[0]
    far = UGV_SENSOR.gamma_at([[4.0, 6.0]], (0.0, 6.0))[0]
    assert near > far > 0.0


def test_rate_peaks_at_rho_under_the_sensor():
    assert UGV_SENSOR.gamma_at([[0.0, 6.0]], (0.0, 6.0))[0] == pytest.approx(UGV_SENSOR.rho)


def test_ugv_is_occluded_where_uav_is_not():
    """block_1 spans x in [-0.75, 0.75], y in [-2, 2]. Look straight through it."""
    observer = (-3.0, 0.0)
    behind = [[3.0, 0.0]]
    assert UGV_SENSOR.gamma_at(behind, observer)[0] == 0.0
    assert UAV_SENSOR.gamma_at(behind, observer)[0] > 0.0


def test_clear_line_of_sight_is_not_blocked():
    observer = (-3.0, 7.0)
    assert UGV_SENSOR.gamma_at([[3.0, 7.0]], observer)[0] > 0.0


def test_grid_and_point_evaluations_agree():
    grid_x, grid_y = np.meshgrid(np.linspace(-8, 8, 33), np.linspace(-6, 6, 25))
    pose = (2.0, 3.0)
    grid = UGV_SENSOR.gamma_grid(grid_x, grid_y, pose)
    pts = UGV_SENSOR.gamma_at(np.column_stack([grid_x.ravel(), grid_y.ravel()]), pose)
    assert np.allclose(grid.ravel(), pts)


def test_detection_probability_is_bounded_and_grows_with_dt():
    pts = [[0.5, 6.0]]
    short = UGV_SENSOR.detection_probability(pts, (0.0, 6.0), 0.1)[0]
    long = UGV_SENSOR.detection_probability(pts, (0.0, 6.0), 5.0)[0]
    assert 0.0 < short < long < 1.0


def test_team_rate_is_additive():
    grid_x, grid_y = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
    poses = [(UAV_SENSOR, (0.0, 0.0)), (UGV_SENSOR, (2.0, 2.0))]
    total = gamma_sum(poses, grid_x, grid_y)
    parts = sum(m.gamma_grid(grid_x, grid_y, p) for m, p in poses)
    assert np.allclose(total, parts)
    assert np.all(total >= 0.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        SensorModel(rho=0.0, sigma=1.0, occluded=False)
    with pytest.raises(ValueError):
        SensorModel(rho=1.0, sigma=-1.0, occluded=False)
    with pytest.raises(ValueError):
        UGV_SENSOR.detection_probability([[0.0, 0.0]], (0.0, 0.0), 0.0)
