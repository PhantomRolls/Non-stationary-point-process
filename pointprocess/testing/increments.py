from pointprocess.testing.transformations import build_eta_on_grid, transform_T_eta_univariate, increments_from_Zhat
from pointprocess.estimation.compensators.exp import compensator_exp
from pointprocess.estimation.compensators.pl import compensator_pl
from pointprocess.estimation.compensators.multiexp import compensator_multiexp
import numpy as np

def simple_compensator_test(events, estimated_params, T, H0):
    """
    Simple compensator-based test: compute inter-arrival times in compensated time
    and test if they are i.i.d. Exp(1).
    
    Under H0, the compensated times Λ(t_i) should form a unit-rate Poisson process,
    so the differences Λ(t_i) - Λ(t_{i-1}) should be i.i.d. Exp(1).
    
    Parameters
    ----------
    events : 1D array_like
        Sorted event times in [0, T].
    estimated_params : dict
        Estimated Hawkes parameters.
    T : float
        Observation horizon.
    H0 : str
        Model type ("exp", "pl", "multiexp", "multiexp-fixed-betas").
    
    Returns
    -------
    compensated_diffs : 1D np.ndarray
        Differences in compensated time (should be Exp(1) if model fits).
    """
    events = np.asarray(events, dtype=float)
    
    # Compute compensator at event times
    if H0 == "exp":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_at_events = compensator_exp(mu, alpha, beta, events, events)
    elif H0 == "pl":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_at_events = compensator_pl(mu, alpha, beta, events, events)
    elif H0 == "multiexp" or H0 == "multiexp-fixed-betas":
        mu = estimated_params["mu"]
        alphas = estimated_params["alphas"]
        betas = estimated_params["betas"]
        Lambda_at_events = compensator_multiexp(mu, alphas, betas, events, events)
    else:
        raise ValueError(f"Unknown H0 model type: {H0}")
    
    # Compute inter-arrival times in compensated time
    compensated_diffs = np.diff(Lambda_at_events)
    
    return compensated_diffs

def naive_increments(events, estimated_params, T, n, tau, grid_size, H0):

    # Grid u in [0, tau]
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, estimated_params, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Standardized empirical process W̃(u)
    mean_intensity_hat = len(events) / T
    Wtilde = eta_tau / np.sqrt(mean_intensity_hat)

    # Increments Z̃_i (eq. 24)
    Ztilde = increments_from_Zhat(Wtilde, grid_u, n, tau=tau)
    return Ztilde


def khmaladze_increments(events, estimated_params, T, n, tau, grid_size, H0):

    # Grid u
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, estimated_params, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Khmaladze transform Ŵ(u)
    mean_intensity_hat = len(events) / T
    What = transform_T_eta_univariate(eta_tau, grid_u, mean_intensity_hat)

    # Increments Ẑ_i
    Zhat = increments_from_Zhat(What, grid_u, n, tau=tau)
    return Zhat
