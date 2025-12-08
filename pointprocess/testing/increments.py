# pointprocess/testing/increments.py
import numpy as np
from pointprocess.testing.transformations import (
    build_eta_on_grid,
    transform_T_eta_univariate,
)

def W_naive(eta_tau, mean_intensity):
    """ Naive standardized process W̃(u). """
    return eta_tau / np.sqrt(mean_intensity)

def W_khmaladze(eta_tau, grid_u, mean_intensity):
    """ Khmaladze-transformed process Ŵ(u). """
    return transform_T_eta_univariate(eta_tau, grid_u, mean_intensity)

def increments_from_W(W_grid, grid_u, n, tau):
    """
    Generic increment builder : 
    Z_i = sqrt(n/tau) * (W(u_i) - W(u_{i-1}))
    """
    W_grid = np.asarray(W_grid, dtype=float)
    grid_u = np.asarray(grid_u, dtype=float)

    # u_i = i * tau / n
    us = np.linspace(0.0, tau, n+1)

    # W(u_i)
    W_interp = np.interp(us, grid_u, W_grid)

    diffs = np.diff(W_interp)
    return np.sqrt(n / tau) * diffs

def build_increments(events, estimated_params, T, n, tau, grid_size, H0, method):
    """
    Compute increments Z_i for either the naïve or Khmaladze method.
    """

    # Build grid
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical compensated process η̂(u)
    eta_full, _, _ = build_eta_on_grid(events, estimated_params, T, grid_u_full, H0)
    eta_tau = eta_full[mask]

    # Mean intensity  μ̄ = N(T)/T  (correct! Not baseline mu)
    mean_intensity = len(events) / T

    # Select transformation
    if method == "naive":
        W = W_naive(eta_tau, mean_intensity)

    elif method == "khmaladze":
        W = W_khmaladze(eta_tau, grid_u, mean_intensity)

    else:
        raise ValueError("method must be 'naive' or 'khmaladze'")

    # Build increments (same for both)
    Z = increments_from_W(W, grid_u, n, tau)
    return Z
