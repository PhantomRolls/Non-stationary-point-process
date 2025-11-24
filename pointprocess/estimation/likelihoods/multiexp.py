import numpy as np
from numba import njit

@njit(fastmath=True)
def hawkes_multiexp_loglik(mu, alphas, betas, events, T, dt, tail):
    n = events.size
    J = alphas.size
    
    if mu <= 0 or np.any(alphas < 0) or np.any(betas <= 0):
        return -1e20
    
    if n == 0:
        return -mu * T
    
    # R recursion: shape (J, n)
    R = np.zeros((J, n))
    
    for j in range(J):
        for k in range(1, n):
            R[j, k] = np.exp(-betas[j] * dt[k-1]) * (1.0 + R[j, k-1])
    
    lam_log_sum = 0.0
    for k in range(n):
        lam = mu
        for j in range(J):
            lam += alphas[j] * R[j, k]
        
        if lam <= 0:
            return -1e20
        
        lam_log_sum += np.log(lam)
    
    # Integral
    integral = mu * T
    for j in range(J):
        integral += (alphas[j] / betas[j]) * np.sum(1.0 - np.exp(-betas[j] * tail))
    
    return lam_log_sum - integral
