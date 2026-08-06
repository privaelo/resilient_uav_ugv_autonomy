from ipp_core.belief_field import BeliefField
from ipp_core.intensity import Component, IntensityField
from ipp_core.sensing import gamma_sum, UAV_SENSOR, UGV_SENSOR

import numpy as np
import pytest


def make_field(seed=1, resolution=0.5):
    field = IntensityField.from_seed(seed, resolution=resolution)
    return field, BeliefField(field.evaluate(), field.cell_area)


def test_starts_at_prior_mass():
    field, belief = make_field()
    assert belief.undetected_mass() == pytest.approx(field.total_mass())
    assert belief.fraction_resolved() == pytest.approx(0.0)
    assert belief.j == 0.0


def test_thinning_never_increases_lambda():
    field, belief = make_field()
    prior = belief.lam.copy()
    rate = gamma_sum([(UGV_SENSOR, (2.0, 1.0))], field.grid_x, field.grid_y)
    belief.step(rate, 0.5)
    assert np.all(belief.lam <= prior + 1e-12)


def test_undetected_mass_is_monotone_under_arbitrary_motion():
    """The regression guard. The original product form violated this."""
    field, belief = make_field()
    rng = np.random.default_rng(7)
    previous = belief.undetected_mass()
    for _ in range(40):
        pose = (rng.uniform(-14, 14), rng.uniform(-9, 9))
        rate = gamma_sum([(UGV_SENSOR, pose), (UAV_SENSOR, pose)], field.grid_x, field.grid_y)
        current = belief.step(rate, 0.1)
        assert current <= previous + 1e-12
        previous = current


def test_loitering_yields_diminishing_returns():
    """A stationary robot's marginal reduction must decay monotonically toward zero."""
    field, belief = make_field()
    rate = gamma_sum([(UGV_SENSOR, (0.0, 6.0))], field.grid_x, field.grid_y)

    gains = []
    for _ in range(45):
        before = belief.undetected_mass()
        belief.step(rate, 0.5)
        gains.append(before - belief.undetected_mass())

    assert gains[0] > 0.0
    # Guaranteed: every cell's contribution decays geometrically, so the sum does too.
    assert all(b <= a + 1e-12 for a, b in zip(gains, gains[1:]))
    # Cells far from the sensor have tiny gamma and drain slowly, so the tail is not
    # arbitrarily small -- but it is a small fraction of the opening step.
    assert gains[-1] < gains[0] * 0.05


def test_j_accumulates_monotonically():
    field, belief = make_field()
    rate = gamma_sum([(UGV_SENSOR, (0.0, 0.0))], field.grid_x, field.grid_y)
    seen = [belief.j]
    for _ in range(10):
        belief.step(rate, 0.2)
        seen.append(belief.j)
    assert seen[0] == 0.0
    assert all(b > a for a, b in zip(seen, seen[1:]))
    assert belief.elapsed == pytest.approx(2.0)


def test_platform_advantage_depends_on_where_the_mass_is():
    """Neither platform dominates -- which is the reason to carry both."""
    def run(sensor, field, poses, dt=0.2):
        belief = BeliefField(field.evaluate(), field.cell_area)
        for pose in poses:
            belief.step(gamma_sum([(sensor, pose)], field.grid_x, field.grid_y), dt)
        return belief.j

    # Mass concentrated tightly under the path: high rho wins.
    tight = IntensityField([Component(0.0, 7.0, 0.8, 6.0)], resolution=0.25)
    dwell = [(0.0, 7.0)] * 30
    assert run(UGV_SENSOR, tight, dwell) < run(UAV_SENSOR, tight, dwell)

    # Mass spread far off the path: the wide footprint wins despite lower rho.
    spread = IntensityField([Component(0.0, 7.0, 3.5, 6.0)], resolution=0.25)
    sweep = [(x, 0.0) for x in np.linspace(-10.0, 10.0, 30)]
    assert run(UAV_SENSOR, spread, sweep) < run(UGV_SENSOR, spread, sweep)


def test_resolve_zeroes_mass_locally():
    field, belief = make_field()
    before = belief.undetected_mass()
    centre = (field.components[0].x, field.components[0].y)
    belief.resolve([centre], 1.0, field.grid_x, field.grid_y)
    assert belief.undetected_mass() < before


def test_discretization_converges():
    """U(0) must not depend on how finely the arena is cut."""
    masses = [IntensityField.from_seed(3, resolution=r).total_mass()
              for r in (0.5, 0.25, 0.125)]
    assert masses[1] == pytest.approx(masses[0], rel=0.05)
    assert masses[2] == pytest.approx(masses[1], rel=0.02)


def test_reset_restores_prior():
    field, belief = make_field()
    rate = gamma_sum([(UGV_SENSOR, (0.0, 0.0))], field.grid_x, field.grid_y)
    belief.step(rate, 1.0)
    belief.reset()
    assert belief.undetected_mass() == pytest.approx(belief.initial_mass())
    assert belief.j == 0.0


def test_rejects_bad_input():
    field, belief = make_field()
    with pytest.raises(ValueError):
        belief.step(np.zeros(belief.shape), 0.0)
    with pytest.raises(ValueError):
        belief.step(np.zeros((2, 2)), 0.1)
    with pytest.raises(ValueError):
        belief.step(-np.ones(belief.shape), 0.1)
    with pytest.raises(ValueError):
        BeliefField(field.evaluate(), 0.0)
