from ipp_core.geometry import inside_obstacle
from ipp_core.intensity import Component, IntensityField

import numpy as np
import pytest


def test_same_seed_gives_identical_field():
    a = IntensityField.from_seed(42)
    b = IntensityField.from_seed(42)
    assert a.components == b.components
    assert np.array_equal(a.evaluate(), b.evaluate())


def test_different_seed_gives_different_field():
    a = IntensityField.from_seed(1)
    b = IntensityField.from_seed(2)
    assert a.components != b.components


def test_same_seed_gives_identical_target_realization():
    field = IntensityField.from_seed(11)
    first = field.sample_targets(np.random.default_rng(5))
    second = field.sample_targets(np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_expected_count_matches_integrated_intensity():
    """N ~ Poisson(integral lambda_0), so the sample mean must track total_mass."""
    field = IntensityField([Component(0.0, 0.0, 2.0, 5.0)], resolution=0.1)
    expected = field.total_mass()
    counts = [len(field.sample_targets(np.random.default_rng(s))) for s in range(300)]
    assert np.mean(counts) == pytest.approx(expected, rel=0.15)


def test_components_are_placed_clear_of_obstacles():
    for seed in range(20):
        field = IntensityField.from_seed(seed)
        pts = [[c.x, c.y] for c in field.components]
        assert not inside_obstacle(pts).any()


def test_targets_never_spawn_inside_obstacles():
    field = IntensityField.from_seed(3)
    targets = field.sample_targets(np.random.default_rng(0))
    if len(targets):
        assert not inside_obstacle(targets).any()


def test_total_mass_scales_with_weight():
    light = IntensityField([Component(0.0, 0.0, 2.0, 1.0)], resolution=0.1)
    heavy = IntensityField([Component(0.0, 0.0, 2.0, 4.0)], resolution=0.1)
    assert heavy.total_mass() == pytest.approx(4.0 * light.total_mass(), rel=1e-6)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        IntensityField([], resolution=0.25)
    with pytest.raises(ValueError):
        IntensityField([Component(0.0, 0.0, 1.0, 1.0)], resolution=0.0)
