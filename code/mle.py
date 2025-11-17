import numpy as np
from scipy.optimize import minimize
from numba import njit


@njit
def hawkes_loglik(params, events, T):
    """
    Log-likelihood of a univariate Hawkes process with kernel
    phi(t) = alpha * exp(-beta * t), t > 0
    and intensity lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t - t_i)).
    """
    mu, alpha, beta = params

    # Parameter constraints
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -1e20   # big negative penalty instead of -inf

    n = events.size
    if n == 0:
        # No events: likelihood reduces to exp(-mu T)
        return -mu * T

    # Recursion for R_i = sum_{j<i} exp(-beta (t_i - t_j))
    R = np.empty(n)
    R[0] = 0.0  # no past events before the first one

    if n > 1:
        dt = np.diff(events)
        e = np.exp(-beta * dt)
        for i in range(1, n):
            R[i] = e[i-1] * (1.0 + R[i-1])

    # Intensity at event times
    lam = mu + alpha * R
    if np.any(lam <= 0):
        return -1e20

    term1 = np.log(lam).sum()

    # Integral of the kernel from each event to T
    tail = T - events
    term2 = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * tail))

    return term1 - term2


@njit
def hawkes_pl_loglik_numba_window(mu, alpha, beta, events, T, L):
    n = events.size

    if mu <= 0 or alpha < 0 or beta <= 0:
        return -1e20
    if n == 0:
        return -mu * T

    lam_log_sum = 0.0
    j0 = 0  # index du premier événement encore dans la fenêtre

    for k in range(n):
        t_k = events[k]

        # Avancer j0 tant que l'événement est trop vieux
        while j0 < k and t_k - events[j0] > L:
            j0 += 1

        # Somme des contributions seulement sur [j0, k-1]
        kernel_sum = 0.0
        for i in range(j0, k):
            dt = t_k - events[i]
            kernel_sum += (1.0 + dt) ** (-(beta + 1.0))

        lam_k = mu + alpha * kernel_sum
        if lam_k <= 0:
            return -1e20
        lam_log_sum += np.log(lam_k)

    # Terme intégral (O(n), pas un souci)
    tail = T - events
    term2 = mu * T + (alpha / beta) * np.sum(1.0 - (1.0 + tail) ** (-beta))

    return lam_log_sum - term2


def fit_hawkes(events, T, H0, x0=(0.5, 0.8, 1.0), L=None):
    events = np.asarray(events, dtype=np.float64)

    if L is None:
        # rough rule of thumb: beta-adaptive window
        # you can tune it more precisely depending on your application
        L = T / 10.0
        
    def obj(p):
        if H0 == "exp":
            return -hawkes_loglik(p, events, T)
        elif H0 == "pl":
            return -hawkes_pl_loglik_numba_window(p[0], p[1], p[2], events, T, L)

    bounds = [
        (1e-8, None),   # mu > 0
        (0.0, None),    # alpha >= 0
        (1e-8, None)    # beta > 0
    ]

    res = minimize(
        obj,
        x0=x0,
        method="L-BFGS-B",   # good for bounded smooth problems
        bounds=bounds,
        options={"maxiter": 1000}
    )
    return res
