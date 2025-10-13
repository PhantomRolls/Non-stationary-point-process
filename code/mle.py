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

def self_correcting_loglik(params, events, T):
    mu, alpha, beta = params
    if beta <= 0:
        return -np.inf
    n = len(events)
    if n == 0:
        return -mu*T
    
    log_intensities = []
    integral = 0.0
    
    for i, ti in enumerate(events):
        exponent = mu + beta * ti - alpha * i
        exponent = np.clip(exponent, -700, 709)
        log_intensities.append(exponent)
        
        if i == 0:
            ti_prev = 0
        else:
            ti_prev = events[i-1]
        
        if abs(beta) > 1e-8:
            integral += (np.exp(mu - alpha * i) / beta) * (np.exp(beta * ti) - np.exp(beta * ti_prev))
        else:
            integral += np.exp(mu - alpha * i) * (ti - ti_prev)
    
    if abs(beta) > 1e-8:
        integral += (np.exp(mu - alpha * n) / beta) * (np.exp(beta * T) - np.exp(beta * events[-1]))
    else:
        integral += np.exp(mu - alpha * n) * (T - events[-1])
    
    loglik = np.sum(log_intensities) - integral
    return loglik

def fit_self_correcting(events, T):
    obj = lambda p: -self_correcting_loglik(p, events, T)
    res = minimize(obj, x0=[0.5, 0.5, 0.1], bounds=[(None,None),(None,None),(1e-8,None)])
    return res

