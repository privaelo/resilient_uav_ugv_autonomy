"""
Per-platform detection rates.

A robot at pose x detects a target at l with instantaneous rate

    gamma_i(l ; x_i) = rho_i * g_i(l ; x_i) * visible_i(l, x_i)

rho is the platform's maximum detection capability and g its geometric footprint. The
UAV overflies obstacles and always has line of sight; the UGV does not, and that
asymmetry is the reason dispatching a ground robot into a swept region can still pay.
"""

from dataclasses import dataclass

from ipp_core.geometry import segment_blocked

import numpy as np


@dataclass(frozen=True)
class SensorModel:
    """Detection-rate model for one platform class."""

    rho: float
    sigma: float
    occluded: bool
    name: str = ''

    def __post_init__(self):
        if self.rho <= 0:
            raise ValueError(f'rho must be positive, got {self.rho}')
        if self.sigma <= 0:
            raise ValueError(f'sigma must be positive, got {self.sigma}')

    def footprint(self, grid_x, grid_y, pose):
        """Return the geometric term g over a grid, ignoring visibility."""
        d2 = (grid_x - pose[0]) ** 2 + (grid_y - pose[1]) ** 2
        return np.exp(-d2 / (2.0 * self.sigma ** 2))

    def gamma_grid(self, grid_x, grid_y, pose):
        """Return the detection rate at every cell center, shape matching ``grid_x``."""
        gamma = self.rho * self.footprint(grid_x, grid_y, pose)
        if not self.occluded:
            return gamma
        pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        blocked = segment_blocked(pose[:2], pts).reshape(grid_x.shape)
        return np.where(blocked, 0.0, gamma)

    def gamma_at(self, points, pose):
        """Return the detection rate at arbitrary points, shape (n,)."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return np.empty(0, dtype=float)
        d2 = (points[:, 0] - pose[0]) ** 2 + (points[:, 1] - pose[1]) ** 2
        gamma = self.rho * np.exp(-d2 / (2.0 * self.sigma ** 2))
        if not self.occluded:
            return gamma
        return np.where(segment_blocked(pose[:2], points), 0.0, gamma)

    def detection_probability(self, points, pose, dt):
        """Probability of at least one detection over a window ``dt``."""
        if dt <= 0:
            raise ValueError(f'dt must be positive, got {dt}')
        return 1.0 - np.exp(-self.gamma_at(points, pose) * dt)


# Wide footprint, low capability -- sees a lot, confirms little, ignores obstacles.
UAV_SENSOR = SensorModel(rho=0.15, sigma=6.0, occluded=False, name='uav')

# Narrow footprint, high capability -- confirms fast, but only what it can see.
UGV_SENSOR = SensorModel(rho=1.20, sigma=1.5, occluded=True, name='ugv')


def gamma_sum(models_and_poses, grid_x, grid_y):
    """Sum detection rates over a team, shape matching ``grid_x``."""
    total = np.zeros(grid_x.shape, dtype=float)
    for model, pose in models_and_poses:
        total += model.gamma_grid(grid_x, grid_y, pose)
    return total
