# pointprocess/estimation/compensators/pl.py
from numba import njit
import numpy as np

@njit(fastmath=True)
def compensator_pl(mu, alpha, beta, events, grid_times):
    """
    Compensator on an arbitrary grid (for the Khmaladze transform).
    Integrates λ(t) = μ + α Σ_{t_i<t} (1 + t - t_i)^{-(β+1)} from 0 to each grid_times[j].

    Returns: Lambda_grid[j] = ∫_0^{grid_times[j]} λ(s) ds
    """
    n_events = len(events)
    n_grid = len(grid_times)
    Lambda = np.zeros(n_grid, dtype=np.float64)

    c = alpha / beta  # constant for the integral

    for j in range(n_grid):
        t = grid_times[j]

        # Baseline part
        Lambda[j] = mu * t

        # Kernel part: for each past event t_i < t
        for i in range(n_events):
            if events[i] >= t:
                break
            dt = t - events[i]
            # Integral of (1+s)^{-(β+1)} from 0 to dt:
            # = [-(1+s)^{-β}/β]_0^dt = (1/β)[1 - (1+dt)^{-β}]
            Lambda[j] += c * (1.0 - (1.0 + dt) ** (-beta))

    return Lambda