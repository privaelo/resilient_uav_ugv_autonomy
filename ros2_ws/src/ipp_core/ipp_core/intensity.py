"""
Latent intensity field and the Poisson target realizations drawn from it.

The team never sees this field. It defines where targets actually are, and the belief
in ``belief_field`` is initialized from a (possibly wrong) prior of the same form.
"""

from dataclasses import dataclass

from ipp_core.geometry import ARENA_BOUNDS, inside_obstacle

import numpy as np


@dataclass(frozen=True)
class Component:
    """One Gaussian bump of target mass."""

    x: float
    y: float
    sigma: float
    weight: float


class IntensityField:
    """An inhomogeneous Poisson intensity over the arena, discretized on a grid."""

    def __init__(self, components, bounds=ARENA_BOUNDS, resolution=0.25):
        if resolution <= 0:
            raise ValueError(f'resolution must be positive, got {resolution}')
        if not components:
            raise ValueError('need at least one component')
        self.components = tuple(components)
        self.bounds = tuple(float(b) for b in bounds)
        self.resolution = float(resolution)

        x_min, x_max, y_min, y_max = self.bounds
        self.xs = np.arange(x_min + resolution / 2, x_max, resolution)
        self.ys = np.arange(y_min + resolution / 2, y_max, resolution)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)
        self.cell_area = self.resolution ** 2

    @classmethod
    def from_seed(cls, seed, n_components=3, bounds=ARENA_BOUNDS, resolution=0.25,
                  sigma_range=(1.5, 3.5), weight_range=(2.0, 6.0), margin=1.5):
        """Draw a mixture whose component placement is fully determined by ``seed``."""
        rng = np.random.default_rng(seed)
        x_min, x_max, y_min, y_max = bounds
        components = []
        while len(components) < n_components:
            x = rng.uniform(x_min + margin, x_max - margin)
            y = rng.uniform(y_min + margin, y_max - margin)
            if inside_obstacle([[x, y]], inflate=margin)[0]:
                continue
            components.append(Component(
                x=x,
                y=y,
                sigma=rng.uniform(*sigma_range),
                weight=rng.uniform(*weight_range),
            ))
        return cls(components, bounds=bounds, resolution=resolution)

    @property
    def shape(self):
        return self.grid_x.shape

    def evaluate(self):
        """Return lambda_0 sampled at every cell center, shape (H, W)."""
        lam = np.zeros(self.shape, dtype=float)
        for c in self.components:
            d2 = (self.grid_x - c.x) ** 2 + (self.grid_y - c.y) ** 2
            lam += c.weight * np.exp(-d2 / (2.0 * c.sigma ** 2)) / (2.0 * np.pi * c.sigma ** 2)
        return lam

    def total_mass(self):
        """Return the expected target count, the integral of lambda_0 over the arena."""
        return float(self.evaluate().sum() * self.cell_area)

    def sample_targets(self, rng):
        """
        Draw one Poisson realization: N ~ Poisson(total_mass), positions ~ lambda_0.

        Returns an (N, 2) array. Positions are drawn by rejection sampling, so the same
        generator state always produces the same realization.
        """
        lam = self.evaluate()
        n = int(rng.poisson(self.total_mass()))
        if n == 0:
            return np.empty((0, 2), dtype=float)

        x_min, x_max, y_min, y_max = self.bounds
        ceiling = float(lam.max())
        out = []
        while len(out) < n:
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if inside_obstacle([[x, y]])[0]:
                continue
            density = 0.0
            for c in self.components:
                d2 = (x - c.x) ** 2 + (y - c.y) ** 2
                density += c.weight * np.exp(-d2 / (2.0 * c.sigma ** 2)) / (
                    2.0 * np.pi * c.sigma ** 2)
            if rng.uniform(0.0, ceiling) <= density:
                out.append((x, y))
        return np.asarray(out, dtype=float)
