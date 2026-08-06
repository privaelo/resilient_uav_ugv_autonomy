from ipp_core.geometry import inside_obstacle, obstacle_bounds, OBSTACLES, segment_blocked

import numpy as np


def test_bounds_match_declared_boxes():
    lo, hi = obstacle_bounds()
    assert lo.shape == (len(OBSTACLES), 2)
    # block_1 is a 1.5 x 4.0 box centred on the origin.
    assert np.allclose(lo[0], (-0.75, -2.0))
    assert np.allclose(hi[0], (0.75, 2.0))


def test_inflation_grows_boxes():
    lo, hi = obstacle_bounds(inflate=0.5)
    assert np.allclose(lo[0], (-1.25, -2.5))
    assert np.allclose(hi[0], (1.25, 2.5))


def test_view_through_a_block_is_blocked():
    assert segment_blocked((-3.0, 0.0), [[3.0, 0.0]])[0]


def test_view_around_a_block_is_clear():
    assert not segment_blocked((-3.0, 7.0), [[3.0, 7.0]])[0]


def test_view_stops_short_of_a_block():
    """The segment ends before reaching block_1, so nothing is occluded."""
    assert not segment_blocked((-5.0, 0.0), [[-2.0, 0.0]])[0]


def test_axis_parallel_view_is_handled():
    # Straight up the y axis, directly through block_1.
    assert segment_blocked((0.0, -5.0), [[0.0, 5.0]])[0]
    # Straight up the y axis, well clear of every block.
    assert not segment_blocked((12.0, -5.0), [[12.0, 5.0]])[0]


def test_batch_evaluation_matches_individual():
    origin = (-4.0, 1.0)
    pts = [[3.0, 0.0], [3.0, 7.0], [-6.0, -3.0], [10.0, 5.0]]
    batch = segment_blocked(origin, pts)
    one_by_one = [segment_blocked(origin, [p])[0] for p in pts]
    assert list(batch) == one_by_one


def test_inside_obstacle_detects_containment():
    assert inside_obstacle([[0.0, 0.0]])[0]
    assert not inside_obstacle([[8.0, 7.0]])[0]


def test_barriers_do_not_occlude_interior_views():
    """Walls are in OBSTACLES for containment but must not block interior sight lines."""
    assert not segment_blocked((-13.0, 8.0), [[13.0, 8.0]])[0]
