import numpy as np
from numba import njit



# -------------------------------------------------------
# 1) Compensator for the naive methode
# -------------------------------------------------------
def compensator_exp_fast(mu, alpha, beta, events):
    events = np.asarray(events)
    n = events.size
    if n == 0:
        return np.array([])

    Lambda = np.zeros(n, dtype=float)
    prev_time = 0.0
    Lambda_prev = 0.0
    S_prev = 0.0  # S(t) = sum alpha * exp(-beta(t - t_j))

    for i, t in enumerate(events):
        Delta = t - prev_time

        base_integral = mu * Delta
        excitation_integral = (S_prev / beta) * (1.0 - np.exp(-beta * Delta))

        Lambda_i = Lambda_prev + base_integral + excitation_integral
        Lambda[i] = Lambda_i

        S_prev = S_prev * np.exp(-beta * Delta) + alpha

        prev_time = t
        Lambda_prev = Lambda_i

    return Lambda

@njit
def compensator_pl_exact(mu, alpha, beta, events):
    """
    Exact compensator for a Hawkes process with power-law kernel:
        phi(t) = (1 + t)^(-(beta + 1)), t > 0

    Lambda(t_k) = mu * t_k
                  + (alpha / beta) * sum_{i < k} [1 - (1 + t_k - t_i)^(-beta)]
    """
    n = events.size
    Lambda = np.empty(n, dtype=np.float64)
    c = alpha / beta

    for k in range(n):
        t_k = events[k]
        if k == 0:
            # no past events
            Lambda[k] = mu * t_k
        else:
            s = 0.0
            for i in range(k):
                dt = t_k - events[i]
                s += 1.0 - (1.0 + dt) ** (-beta)
            Lambda[k] = mu * t_k + c * s

    return Lambda



# -------------------------------------------------------
# 2) Compensator on a time grid for the Khmaladze transform
# -------------------------------------------------------
@njit
def compensator_pl_on_grid(mu, alpha, beta, events, grid_times):
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


def compensator_exp_on_grid(mu, alpha, beta, events, grid_times):
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