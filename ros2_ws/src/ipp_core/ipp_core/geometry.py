"""
Arena geometry shared by the sensing model, the belief field, and the UGV controller.

These boxes mirror ``multi_robot_bringup/worlds/urban_obstacles.sdf``. Keep them in sync
when the world changes -- this module is the single definition everything else reads.
"""

import numpy as np

# (center_x, center_y, half_x, half_y) for each static box in the world.
OBSTACLES = (
    (0.0, 0.0, 0.75, 2.00),     # block_1
    (5.0, 4.0, 1.00, 1.50),     # block_2
    (6.0, -4.0, 1.00, 1.50),    # block_3
    (-5.0, 3.0, 1.25, 1.25),    # block_4
    (-6.0, -3.0, 1.25, 1.25),   # block_5
    (0.0, 10.0, 15.00, 0.30),   # barrier_north
    (0.0, -10.0, 15.00, 0.30),  # barrier_south
    (-15.0, 0.0, 0.30, 10.00),  # barrier_west
    (15.0, 0.0, 0.30, 10.00),   # barrier_east
)

# Interior of the walled arena, as (x_min, x_max, y_min, y_max).
ARENA_BOUNDS = (-14.5, 14.5, -9.5, 9.5)

# Blocks only -- the barriers are the arena walls and never occlude an interior view.
_BLOCKS = OBSTACLES[:5]


def obstacle_bounds(obstacles=None, inflate=0.0):
    """Return obstacle extents as ``(lo, hi)`` arrays of shape (n, 2)."""
    obstacles = OBSTACLES if obstacles is None else obstacles
    boxes = np.asarray(obstacles, dtype=float)
    centers = boxes[:, :2]
    halves = boxes[:, 2:] + inflate
    return centers - halves, centers + halves


def segment_blocked(origin, targets, obstacles=None):
    """
    Test line of sight from ``origin`` to each point in ``targets``.

    Uses the slab method against axis-aligned boxes. ``targets`` is an (n, 2) array;
    the return is a boolean array of shape (n,) that is True where the view is blocked.
    """
    origin = np.asarray(origin, dtype=float).reshape(2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    lo, hi = obstacle_bounds(_BLOCKS if obstacles is None else obstacles)

    # Broadcast to (n_targets, n_boxes, 2).
    d = (targets - origin)[:, None, :]
    p0 = origin[None, None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (lo[None, :, :] - p0) * inv
        t2 = (hi[None, :, :] - p0) * inv

    # A zero component means the segment is parallel to that axis. Such an axis cannot
    # constrain the crossing interval, so widen it to (-inf, inf); if the origin is also
    # outside the slab the box is unreachable, which min/max cannot express and so is
    # carried separately.
    parallel = d == 0.0
    misses = parallel & ((p0 < lo[None, :, :]) | (p0 > hi[None, :, :]))
    t1 = np.where(parallel, -np.inf, t1)
    t2 = np.where(parallel, np.inf, t2)

    t_near = np.maximum(np.minimum(t1, t2).max(axis=2), 0.0)
    t_far = np.minimum(np.maximum(t1, t2).min(axis=2), 1.0)
    hit = (t_near <= t_far) & ~np.any(misses, axis=2)
    return np.any(hit, axis=1)


def inside_obstacle(points, inflate=0.0, obstacles=None):
    """Return a boolean mask marking points that fall inside any box."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    lo, hi = obstacle_bounds(obstacles, inflate=inflate)
    within = (points[:, None, :] >= lo[None, :, :]) & (points[:, None, :] <= hi[None, :, :])
    return np.any(np.all(within, axis=2), axis=1)
