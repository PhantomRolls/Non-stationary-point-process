import numpy as np
from scipy.stats import kstest, anderson, cramervonmises, norm
import Hawkes as hk
import time

# -------------------------------------------------------
# 1) Compensator on a time grid for an exponential Hawkes
# -------------------------------------------------------
def Lambda_on_grid_exp(mu, alpha, beta, events, grid_times):
    """
    Compute the compensator
        Lambda_hat(t) = ∫_0^t lambda_hat(s) ds
    for t in grid_times (sorted array).

    Model: univariate exponential Hawkes process with intensity
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

    Parameters
    ----------
    mu : float
        Baseline intensity parameter.
    alpha : float
        Excitation jump size.
    beta : float
        Decay rate (> 0).
    events : 1D array_like
        Sorted event times t_i in [0, T].
    grid_times : 1D array_like
        Sorted times in [0, T] at which Lambda_hat(t) is evaluated.

    Returns
    -------
    Lambda : 1D np.ndarray
        Compensator values at each grid time.
    """
    events = np.asarray(events, dtype=float)
    grid_times = np.asarray(grid_times, dtype=float)
    n_grid = grid_times.size

    Lambda = np.zeros(n_grid, dtype=float)

    # Recursive state:
    prev_time = 0.0
    Lambda_prev = 0.0
    # S(t) = sum_j alpha * exp(-beta * (t - t_j)) at current time t
    S_prev = 0.0
    event_idx = 0
    n_events = events.size

    for i, t in enumerate(grid_times):
        # 1) Process all events that occur between prev_time and t
        while event_idx < n_events and events[event_idx] <= t:
            te = events[event_idx]
            # integrate from prev_time up to event time te
            dt = te - prev_time
            if dt > 0:
                base_int = mu * dt
                # CHANGED: divide by beta in the excitation integral
                excite_int = S_prev * (1.0 - np.exp(-beta * dt)) / beta
                Lambda_prev += base_int + excite_int
                # decay S to time te
                S_prev = S_prev * np.exp(-beta * dt)

            # at the event time te, S jumps by +alpha
            S_prev += alpha
            prev_time = te
            event_idx += 1

        # 2) Integrate from prev_time to current grid time t
        dt = t - prev_time
        if dt > 0:
            base_int = mu * dt
            # CHANGED: divide by beta here as well
            excite_int = S_prev * (1.0 - np.exp(-beta * dt)) / beta
            Lambda_prev += base_int + excite_int
            # decay S to time t
            S_prev = S_prev * np.exp(-beta * dt)
            prev_time = t

        Lambda[i] = Lambda_prev

    return Lambda


# -------------------------------------------------------
# 2) Build eta^{(T)}(u) on a grid
# -------------------------------------------------------
def build_eta_on_grid(events, mu_hat, alpha_hat, beta_hat, T, grid_u):
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

    Lambda_grid = Lambda_on_grid_exp(mu_hat, alpha_hat, beta_hat, events, grid_times)

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
