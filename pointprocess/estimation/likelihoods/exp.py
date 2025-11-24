import numpy as np
from numba import njit

@njit(fastmath=True)
def hawkes_exp_loglik(params, events, T, dt, tail):
    mu, alpha, beta = params
    
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -1e20
    
    n = events.size
    if n == 0:
        return -mu * T
    
    # R_k recursion
    R = np.zeros(n)
    for k in range(1, n):
        R[k] = np.exp(-beta * dt[k-1]) * (1.0 + R[k-1])
    
    lam_log_sum = 0.0
    for k in range(n):
        lam = mu + alpha * R[k]
        if lam <= 0:
            return -1e20
        lam_log_sum += np.log(lam)
    
    # Integral
    integral = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * tail))
    
    return lam_log_sum - integral
