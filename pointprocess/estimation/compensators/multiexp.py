import numpy as np
from numba import njit

@njit(fastmath=True)
def compensator_multiexp(mu, alphas, betas, events, grid_times):
    """
    Compensator Λ(t) evaluated on grid_times for a multi-exponential Hawkes:
    
        λ(t) = μ + ∑_j α_j ∑_{t_i < t} exp(-β_j (t - t_i))
    
    We compute the compensator via the recursion on S_j(t),
    exactly like in the exponential Hawkes case, but with
    one state S_j for each kernel j.
    """

    events = np.asarray(events, dtype=np.float64)
    grid_times = np.asarray(grid_times, dtype=np.float64)

    n_grid = grid_times.size
    n_events = events.size
    J = alphas.size

    # Compensator stored on the grid
    Lambda = np.zeros(n_grid, dtype=np.float64)

    # Recursive state
    prev_time = 0.0
    Lambda_prev = 0.0
    event_idx = 0

    # S_j(t) for each kernel j
    S = np.zeros(J, dtype=np.float64)

    for g in range(n_grid):
        t = grid_times[g]

        # 1) Process events between prev_time and t
        while event_idx < n_events and events[event_idx] <= t:
            te = events[event_idx]

            dt = te - prev_time
            if dt > 0:
                # baseline part
                base_int = mu * dt

                # excitation part: sum_j S_j * (1 - e^{-β_j dt}) / β_j
                excite_int = 0.0
                for j in range(J):
                    b = betas[j]
                    excite_int += S[j] * (1.0 - np.exp(-b * dt)) / b

                Lambda_prev += base_int + excite_int

                # decay S_j to time te
                for j in range(J):
                    S[j] *= np.exp(-betas[j] * dt)

            # At event time: jump of all S_j
            for j in range(J):
                S[j] += alphas[j]

            prev_time = te
            event_idx += 1

        # 2) Integrate from prev_time to t
        dt = t - prev_time
        if dt > 0:
            base_int = mu * dt
            excite_int = 0.0
            for j in range(J):
                b = betas[j]
                excite_int += S[j] * (1.0 - np.exp(-b * dt)) / b

            Lambda_prev += base_int + excite_int

            # decay S to time t
            for j in range(J):
                S[j] *= np.exp(-betas[j] * dt)

            prev_time = t

        Lambda[g] = Lambda_prev

    return Lambda
