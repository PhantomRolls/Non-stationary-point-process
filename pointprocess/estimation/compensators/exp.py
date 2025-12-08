# pointprocess/estimation/compensators/exp.py
from numba import njit
import numpy as np


def compensator_exp(mu, alpha, beta, events, grid_times):
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