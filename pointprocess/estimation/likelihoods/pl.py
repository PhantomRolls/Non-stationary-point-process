import numpy as np
from numba import njit

@njit(fastmath=True)
def hawkes_pl_loglik(mu, alpha, beta, events, T, L, dt, tail):
    n = events.size
    
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -1e20
    
    if n == 0:
        return -mu * T
    
    lam_log_sum = 0.0
    j0 = 0

    for k in range(n):
        t_k = events[k]
        
        while j0 < k and t_k - events[j0] > L:
            j0 += 1
        
        # Sum_{i=j0..k-1} (1 + t_k - t_i)^(-(beta+1))
        kernel_sum = 0.0
        for i in range(j0, k):
            dt_ik = t_k - events[i]
            kernel_sum += (1.0 + dt_ik) ** (-(beta + 1.0))
        
        lam = mu + alpha * kernel_sum
        if lam <= 0:
            return -1e20
        
        lam_log_sum += np.log(lam)
    
    # Integral term
    integral = (
        mu * T
        + (alpha / beta) * np.sum(1.0 - (1.0 + tail) ** (-beta))
    )
    
    return lam_log_sum - integral
