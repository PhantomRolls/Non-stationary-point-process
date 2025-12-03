# pointprocess/testing/transformations.py
import numpy as np
from pointprocess.estimation.compensators.exp import compensator_exp
from pointprocess.estimation.compensators.pl import compensator_pl
from pointprocess.estimation.compensators.multiexp import compensator_multiexp

def build_eta_on_grid(events, estimated_params, T, grid_u, H0):
    grid_u = np.asarray(grid_u, dtype=float)

    if H0 == "exp":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_grid = compensator_exp(mu, alpha, beta, events, grid_u * T)
    elif H0 == "pl":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_grid = compensator_pl(mu, alpha, beta, events, grid_u * T)
    elif H0 == "multiexp":
        mu = estimated_params["mu"]
        alphas = estimated_params["alphas"]
        betas = estimated_params["betas"]
        Lambda_grid = compensator_multiexp(mu, alphas, betas, events, grid_u * T)

    # N(uT): counts of events ≤ each grid time (events are sorted)
    counts = np.searchsorted(events, grid_u * T, side="right")

    eta = (counts - Lambda_grid) / np.sqrt(T)
    return eta, counts, Lambda_grid


def transform_T_eta_univariate(eta_grid, grid_u, mean_intensity):
    """
    Univariate Khmaladze transform:

    Ŵ(u) = ( η(u) - ∫₀ᵘ [η(1) - η(v)]/(1-v) dv ) / sqrt(mean_intensity)

    mean_intensity = N(T)/T
    """

    eta_grid = np.asarray(eta_grid, float)
    grid_u   = np.asarray(grid_u, float)

    eta1 = eta_grid[-1]
    integrand = (eta1 - eta_grid) / (1.0 - grid_u + 1e-16)

    I = np.zeros_like(eta_grid)
    du = np.diff(grid_u)

    for i in range(1, len(grid_u)):
        I[i] = I[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * du[i-1]

    return (eta_grid - I) / np.sqrt(mean_intensity)





