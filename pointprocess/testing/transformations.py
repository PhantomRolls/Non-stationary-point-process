import numpy as np
from pointprocess.estimation.compensators.exp import compensator_exp
from pointprocess.estimation.compensators.pl import compensator_pl
from pointprocess.estimation.compensators.multiexp import compensator_multiexp



# -------------------------------------------------------
# 2) Build eta^{(T)}(u) on a grid
# -------------------------------------------------------
def build_eta_on_grid(events, estimated_params, T, grid_u, H0):
    """
    Build eta^{(T)}(u) = 1/sqrt(T) * ( N(uT) - Lambda_hat(uT) ), for u in grid_u.

    Parameters
    ----------
    events : 1D array_like
        Sorted event times in [0, T].
    mu_hat, alpha_hat, beta_hat : floats
        Estimated Hawkes parameters.
    T : float
        Observation horizon.
    grid_u : 1D array_like
        Grid in [0, 1] on which eta^{(T)} is evaluated.

    Returns
    -------
    eta : 1D np.ndarray
        Values of eta^{(T)}(u) on grid_u.
    counts : 1D np.ndarray
        N(uT) counts at each grid point.
    Lambda_grid : 1D np.ndarray
        Compensator values Lambda_hat(uT) at each grid point.
    """
    grid_u = np.asarray(grid_u, dtype=float)
    grid_times = grid_u * T

    if H0 == "exp":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_grid = compensator_exp(mu, alpha, beta, events, grid_times)
    elif H0 == "pl":
        mu = estimated_params["mu"]
        alpha = estimated_params["alpha"]
        beta = estimated_params["beta"]
        Lambda_grid = compensator_pl(mu, alpha, beta, events, grid_times)
    elif H0 == "multiexp" or H0 == "multiexp-fixed-betas":
        mu = estimated_params["mu"]
        alphas = estimated_params["alphas"]
        betas = estimated_params["betas"]
        Lambda_grid = compensator_multiexp(mu, alphas, betas, events, grid_times)
    else:
        raise ValueError(f"Unknown H0 model type: {H0}")

    # N(uT): counts of events ≤ each grid time (events are sorted)
    counts = np.searchsorted(events, grid_times, side="right")

    eta = (counts - Lambda_grid) / np.sqrt(T)
    return eta, counts, Lambda_grid


# -------------------------------------------------------
# 3) Khmaladze-type transformation T_{theta_T} (univariate)
# -------------------------------------------------------
def transform_T_eta_univariate(eta_grid, grid_u, mean_intensity_hat):
    """
    Apply the univariate transformation T_{theta_T} to eta_grid defined on grid_u:

        W_hat(u) = 1/sqrt(mu_bar_hat) * (
                       eta(u) - ∫_0^u [ eta(1) - eta(v) ] / (1 - v) dv
                   )

    where mu_bar_hat ≈ E[N[0,1]] is the *average* intensity, not the Hawkes baseline.

    Parameters
    ----------
    eta_grid : 1D array_like
        Values of eta^{(T)}(u) on grid_u.
    grid_u : 1D array_like
        Increasing grid in [0, 1] (or [0, tau < 1]).
    mean_intensity_hat : float
        Estimate of E[N[0,1]]. In practice one can take N(T)/T.

    Returns
    -------
    W_hat : 1D np.ndarray
        Transformed process values on grid_u.
    """
    eta_grid = np.asarray(eta_grid, dtype=float)
    grid_u = np.asarray(grid_u, dtype=float)

    eta1 = eta_grid[-1]  # eta(1)

    # Integrand: (eta(1) - eta(v)) / (1 - v)
    # We avoid division by zero at v = 1 by adding a tiny epsilon.
    integrand = (eta1 - eta_grid) / (1.0 - grid_u + 1e-16)

    # Cumulative trapezoidal integral I(u) = ∫_0^u integrand(v) dv
    I = np.zeros_like(eta_grid)
    du = np.diff(grid_u)
    for i in range(1, grid_u.size):
        I[i] = I[i-1] + 0.5 * (integrand[i-1] + integrand[i]) * du[i-1]

    # CHANGED: the scaling uses the *mean* intensity, not the Hawkes baseline
    W_hat = (eta_grid - I) / np.sqrt(mean_intensity_hat)
    return W_hat


# -------------------------------------------------------
# 4) Build Z_hat_i increments (step (iv) in the paper)
# -------------------------------------------------------
def increments_from_Zhat(Zhat_grid, grid_u, n, tau=1.0):
    """
    Given Zhat_grid ≈ W_hat(u) on grid_u, construct:

        Z_i := sqrt(n / tau) * [ Zhat(u_i) - Zhat(u_{i-1}) ],  i = 1..n

    where u_i = i * tau / n.

    Parameters
    ----------
    Zhat_grid : 1D array_like
        Values of W_hat(u) on grid_u.
    grid_u : 1D array_like
        Grid in [0, tau] on which Zhat_grid is defined.
    n : int
        Number of increments.
    tau : float, optional
        Upper bound of the interval (default 1.0). In theory, tau < 1.

    Returns
    -------
    Zi : 1D np.ndarray
        The n standardized increments Z_i.
    """
    Zhat_grid = np.asarray(Zhat_grid, dtype=float)
    grid_u = np.asarray(grid_u, dtype=float)

    # Target evaluation points u_i = i * tau / n, i = 0..n
    us = np.linspace(0.0, tau, n + 1)

    # Interpolate Zhat on these points
    Z_interp = np.interp(us, grid_u, Zhat_grid)

    diffs = np.diff(Z_interp)  # length n
    Zi = np.sqrt(n / tau) * diffs
    return Zi
