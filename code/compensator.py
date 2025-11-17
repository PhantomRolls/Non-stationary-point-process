import numpy as np
from numba import njit


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