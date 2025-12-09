from scipy.optimize import minimize
from pointprocess.estimation.likelihoods.exp import hawkes_exp_loglik
from pointprocess.estimation.likelihoods.pl import hawkes_pl_loglik
from pointprocess.estimation.likelihoods.multiexp import hawkes_multiexp_loglik
import numpy as np

def fit_hawkes(events, T, H0, x0=None, beta=None, betas=None):
    events = np.asarray(events, float)
    n = events.size
    
    # precompute expensive terms ONCE
    dt = np.diff(events) if n > 1 else np.zeros(1)
    tail = T - events

    # ---- EXPONENTIAL ----
    if H0 == "exp":
        if x0 is None:
            x0 = np.array([0.5, 0.8, 1.0])
        
        bounds = [(1e-8,None), (0,None), (1e-8,None)]
        
        def obj(p):
            return -hawkes_exp_loglik(p, events, T, dt, tail)
    
    
    # ---- POWER-LAW ----
    elif H0 == "pl":
        L = T / 10
        
        if x0 is None:
            x0 = np.array([0.5, 0.5, 1.5])
        
        bounds = [(1e-8,None), (0,None), (1e-8,None)]
        
        def obj(p):
            return -hawkes_pl_loglik(p[0], p[1], p[2], events, T, L, tail)

    # ---- MULTI-EXP ----
    elif H0 == "multiexp":
        J=3
        
        if x0 is None:
            mu0 = 0.5
            alpha0 = np.full(J, 0.5/J)
            beta0  = np.linspace(0.5, 2.0, J)
            x0 = np.concatenate(([mu0], alpha0, beta0))
        
        bounds = [(1e-8,None)] + [(0,None)]*J + [(1e-8,None)]*J
        
        alpha_idx = slice(1, 1+J)
        beta_idx  = slice(1+J, 1+2*J)
        
        def obj(p):
            mu = p[0]
            alphas = p[alpha_idx]
            betas  = p[beta_idx]
            return -hawkes_multiexp_loglik(mu, alphas, betas, events, T, dt, tail)
    # ---- MultiExp-Fixed-betas ----
    elif H0 == "multiexp-fixed-betas":
        if betas is None:
            raise ValueError("betas must be provided for multiexp-fixed-betas")
        
        betas = np.asarray(betas, float)
        J = len(betas)
        
        if x0 is None:
            mu0 = 0.5
            alpha0 = np.full(J, 0.5/J)
            x0 = np.concatenate(([mu0], alpha0))
        
        bounds = [(1e-8,None)] + [(0,None)]*J
        
        alpha_idx = slice(1, 1+J)
        
        def obj(p):
            mu = p[0]
            alphas = p[alpha_idx]
            return -hawkes_multiexp_loglik(mu, alphas, betas, events, T, dt, tail)
    else:
        raise ValueError("Unknown H0")

    res = minimize(
        obj,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-7}
    )
    
    p = res.x

    if H0 == "exp":
        params = {
            "mu": p[0],
            "alpha": p[1],
            "beta": p[2]
        }

    elif H0 == "pl":
        params = {
            "mu": p[0],
            "alpha": p[1],
            "beta": p[2],
            "L": L
        }

    elif H0 == "multiexp":
        mu = p[0]
        J = (len(p) - 1) // 2
        alphas = p[1:1+J]
        betas  = p[1+J:1+2*J]

        params = {
            "mu": mu,
            "alphas": alphas,
            "betas": betas,
            "J": J
        }
    elif H0 == "multiexp-fixed-betas":
        J = len(betas)
        mu = p[0]
        alphas = p[1:1+J]

        params = {
            "mu": mu,
            "alphas": alphas,
            "betas": betas,
            "J": J
        }
    # attach dict to result object for convenience
    res.params_dict = params

    return res

