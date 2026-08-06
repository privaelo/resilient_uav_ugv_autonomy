"""
Poisson-thinning posterior over undetected targets, and the J it accumulates.

Thinning a Poisson process leaves a Poisson process, so conditioned on no detection at
l through time t the posterior intensity is available in closed form:

    lambda_t(l) = lambda_0(l) * exp( -integral_0^t sum_i gamma_i(l ; x_i(tau)) dtau )

which makes the whole update an elementwise decay. No particle filter, no target list.

    U(t) = integral lambda_t(l) dl
    J    = integral_0^T U(t) dt

U is non-increasing by construction. Re-sweeping a cleared region buys nothing, so there
is no incentive to loiter on high-lambda cells instead of covering ground.
"""

import numpy as np


class BeliefField:
    """Grid-discretized posterior over undetected target mass."""

    def __init__(self, lam0, cell_area):
        lam0 = np.asarray(lam0, dtype=float)
        if lam0.ndim != 2:
            raise ValueError(f'lam0 must be 2-D, got shape {lam0.shape}')
        if np.any(lam0 < 0):
            raise ValueError('lam0 must be non-negative')
        if cell_area <= 0:
            raise ValueError(f'cell_area must be positive, got {cell_area}')

        self.lam0 = lam0
        self.lam = lam0.copy()
        self.cell_area = float(cell_area)
        self.elapsed = 0.0
        self.j = 0.0

    @property
    def shape(self):
        return self.lam.shape

    def undetected_mass(self):
        """Return the current U(t)."""
        return float(self.lam.sum() * self.cell_area)

    def initial_mass(self):
        """U(0), the total mass the team starts out not having observed."""
        return float(self.lam0.sum() * self.cell_area)

    def fraction_resolved(self):
        """Share of the initial mass that has been swept or confirmed."""
        initial = self.initial_mass()
        if initial == 0.0:
            return 1.0
        return 1.0 - self.undetected_mass() / initial

    def step(self, gamma_sum, dt):
        """
        Advance the posterior by ``dt`` under a team detection-rate field.

        Returns the U(t) that holds after the step. J is accumulated with the
        post-step value, which is the conservative choice: it never credits the team
        for mass it has not yet removed.
        """
        if dt <= 0:
            raise ValueError(f'dt must be positive, got {dt}')
        gamma_sum = np.asarray(gamma_sum, dtype=float)
        if gamma_sum.shape != self.lam.shape:
            raise ValueError(
                f'gamma_sum shape {gamma_sum.shape} does not match field {self.lam.shape}')
        if np.any(gamma_sum < 0):
            raise ValueError('gamma_sum must be non-negative')

        self.lam *= np.exp(-dt * gamma_sum)
        self.elapsed += dt
        u = self.undetected_mass()
        self.j += u * dt
        return u

    def resolve(self, points, radius, grid_x, grid_y):
        """Zero the posterior around confirmed detections."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return
        if radius <= 0:
            raise ValueError(f'radius must be positive, got {radius}')
        for px, py in points:
            hit = (grid_x - px) ** 2 + (grid_y - py) ** 2 <= radius ** 2
            self.lam[hit] = 0.0

    def reset(self):
        """Restore the prior and clear the accumulated objective."""
        self.lam = self.lam0.copy()
        self.elapsed = 0.0
        self.j = 0.0
