import numpy as np
from scipy.optimize import minimize

def hawkes_loglik(params, events, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -np.inf
    n = len(events)
    if n == 0:
        return -mu*T
    R = np.zeros(n)
    R[0] = 1
    for i in range(1, n):
        dt = events[i]-events[i-1]
        R[i] = np.exp(-beta*dt) * (1 + R[i-1])
    lam = mu + alpha*R
    if np.any(lam <= 0):
        return -np.inf
    term1 = np.sum(np.log(lam))
    term2 = mu*T + (alpha/beta) * np.sum(1 - np.exp(-beta*(T-events)))
    return term1 - term2

def fit_hawkes(events, T):
    obj = lambda p: -hawkes_loglik(p, events, T)
    res = minimize(obj, x0=[0.5, 0.8, 1.0], bounds=[(1e-8,None),(0,None),(1e-8,None)])
    return res

