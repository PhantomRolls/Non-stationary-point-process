import numpy as np
from scipy.stats import kstest, kstwobign
import pandas as pd
from tabulate import tabulate
from .simulator import Hawkes
from .mle import fit_hawkes
import matplotlib.pyplot as plt
from scipy.stats import expon
from tqdm import tqdm

def random_time_change(events, params):
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    transformed_events = []
    ti_1 = 0
    gi_1 = 0
    for ti in events:
        delta_i = ti - ti_1
        zi = mu*delta_i + (alpha/beta) * gi_1 * (1 - np.exp(-beta*delta_i))
        transformed_events.append(zi)
        gi_1 = np.exp(-beta*delta_i) * gi_1 + 1
        ti_1 = ti
    return np.array(transformed_events)

def ks_test(transformed_events, alpha=0.05):     
    stat, pval = kstest(transformed_events, 'expon')
    rejette = (pval < alpha)
    return rejette, stat
    
def simulation(n_simulations, T, true_params, alpha=0.05, plot_qq=False):
    n_rejections = 0
    stats = []
    for _ in tqdm(range(n_simulations), desc="Traitement"):
        hawkes = Hawkes(T, true_params)
        events = hawkes.events
        res = fit_hawkes(events, T)
        params = {"mu": res.x[0], "alpha": res.x[1], "beta": res.x[2]}
        transformed_events = random_time_change(events, true_params)
        rejet, stat = ks_test(transformed_events, alpha)
        stats.append(stat * np.sqrt(len(events)))
        if rejet:
            n_rejections += 1
    df = pd.DataFrame({
        "n_simulations": [n_simulations],
        "n_rejections": [n_rejections],
        "T": [T],
        "true_params": [true_params]
    }).T
    df.columns = ["KS test"]
    df.name = "KS test"
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=True))
    stats.sort()
    if plot_qq:
        n = len(stats)
        theo_quantiles = [kstwobign.ppf((i+1-0.5)/n) for i in range(n)]
        plt.figure(figsize=(6,6))
        plt.scatter(theo_quantiles, stats, color='blue', s=10)
        plt.plot([0, max(theo_quantiles)], [0, max(theo_quantiles)], 'r--')
        plt.xlabel("Quantiles théoriques")
        plt.ylabel("Quantiles empiriques")
        plt.title("Q-Q plot des statistiques KS")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.axis('equal')
        plt.show()
    

def plot_ccdf(true_params, T):
    """
    Trace la CCDF empirique des résidus transformés (z_i)
    et la CCDF théorique d'une loi Exp(1).
    """
    hawkes = Hawkes(T, true_params)
    res = fit_hawkes(hawkes.events, T)
    params = {"mu": res.x[0], "alpha": res.x[1], "beta": res.x[2]}
    z = random_time_change(hawkes.events, params)
    z_sorted = np.sort(z)
    n = len(z_sorted)
    ccdf_emp = 1.0 - np.arange(1, n+1) / n  # 1 - F_n(z)
    z_theo = np.linspace(0, z_sorted.max()*1.1, 200)
    ccdf_theo = 1.0 - expon.cdf(z_theo, scale=1.0)
    plt.figure(figsize=(7,5))
    plt.step(z_sorted, ccdf_emp, where="post", label="Empirique (résidus $z_i$)")
    plt.plot(z_theo, ccdf_theo, "r--", lw=2, label="Théorique Exp(1)")
    plt.yscale("log")
    plt.xlabel("z")
    plt.ylabel("CCDF = P(Z > z)")
    plt.title("Comparaison CCDF des résidus transformés vs Exp(1)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


# ---------------------------
# 1) Lambda_on_grid_exp (exponential kernel)
# ---------------------------
def Lambda_on_grid_exp(mu, alpha, beta, events, grid_times):
    r"""
    Compute Lambda_hat(t) = \int_0^t lambda_hat(s) ds for t in grid_times (increasing array),
    with kernel phi(s)=alpha*beta*exp(-beta s) (parametrization as in the library).
    Uses a recursive approach (O(n_events + n_grid)).
    - events: sorted array of event times (t_i)
    - grid_times: increasing array of times (values in [0, T])
    Returns: Lambda_grid array (same shape as grid_times)
    """
    events = np.asarray(events)
    grid_times = np.asarray(grid_times)
    n_grid = grid_times.size
    Lambda = np.zeros(n_grid, dtype=float)

    # stato ricorsivo
    prev_time = 0.0
    Lambda_prev = 0.0
    S_prev = 0.0  # S(t) = sum_j alpha * exp(-beta*(t - t_j)) at current time t
    event_idx = 0
    n_events = events.size

    for i, t in enumerate(grid_times):
        Delta = t - prev_time
        # integral between prev_time and t:
        # baseline part:
        base_int = mu * Delta
        # excitation integral: S_prev * (1 - exp(-beta * Delta))
        excite_int = S_prev * (1.0 - np.exp(-beta * Delta))
        Lambda_i = Lambda_prev + base_int + excite_int
        # BUT we must also add contributions from events that occurred between prev_time and t:
        # if events exist in (prev_time, t], we need to account for their immediate contribution to Lambda at t:
        # We'll process events that happened <= t, updating S_prev and Lambda_i accordingly.
        while event_idx < n_events and events[event_idx] <= t:
            te = events[event_idx]
            # integrate from prev_time to te
            d1 = te - prev_time
            # add integral up to te
            add_base = mu * d1
            add_excite = S_prev * (1.0 - np.exp(-beta * d1))
            Lambda_prev = Lambda_prev + add_base + add_excite
            # now an event occurs at te: at that instant S jumps: S <- S*exp(-beta*0)+alpha = S_prev*exp(-beta*d1) + alpha
            S_prev = S_prev * np.exp(-beta * d1) + alpha
            prev_time = te
            # move to next event (but we still need to integrate from te to t after all events in interval processed)
            event_idx += 1
        # after processing events up to t, now integrate from prev_time (maybe last event time or original prev_time) to t
        Delta2 = t - prev_time
        if Delta2 > 0:
            # add baseline and excitation for final segment
            add_base = mu * Delta2
            add_excite = S_prev * (1.0 - np.exp(-beta * Delta2))
            Lambda_prev = Lambda_prev + add_base + add_excite
            S_prev = S_prev * np.exp(-beta * Delta2)  # decay to time t
            prev_time = t
        Lambda[i] = Lambda_prev
        # Lambda_prev kept for next grid point

    return Lambda

# ---------------------------
# 2) build eta^{(T)}(u) on a grid
# ---------------------------
def build_eta_on_grid(events, mu_hat, alpha_hat, beta_hat, T, grid_u):
    r"""
    Build eta^{(T)}(u) = 1/sqrt(T) * ( N(uT) - Lambda_hat(uT) ), for u in grid_u (values in [0,1]).
    Returns: eta_grid (same shape as grid_u)
    """
    grid_times = grid_u * T
    Lambda_grid = Lambda_on_grid_exp(mu_hat, alpha_hat, beta_hat, events, grid_times)
    # N(uT): counts of events <= each grid time
    # since events sorted:
    counts = np.searchsorted(events, grid_times, side='right')
    eta = (counts - Lambda_grid) / np.sqrt(T)
    return eta, counts, Lambda_grid

# ---------------------------
# 3) T_{theta_T} transformation for univariate case (explicit formula from the paper)
# ---------------------------
def transform_T_eta_univariate(eta_grid, grid_u, mu_hat):
    r"""
        Apply the T_{theta_T} transformation to the array eta_grid defined on grid_u.
        Formula (univariate) used in the paper:
            Zhat(u) = (1/sqrt(mu_hat)) * ( eta(u) - \int_0^u [ eta(1) - eta(v) ]/(1-v) dv )
        grid_u: increasing array in [0,1].
        The integral is computed numerically using the trapezoidal rule.
        Returns Zhat_grid (same shape as eta_grid).
        """
    # ensure arrays as numpy
    eta_grid = np.asarray(eta_grid)
    grid_u = np.asarray(grid_u)
    # numeric integral I(u) = \int_0^u (eta(1) - eta(v)) / (1 - v) dv
    eta1 = eta_grid[-1]  # eta(1)
    # avoid dividing by (1-v)=0 at v=1; we will not evaluate integrand at v=1 as grid_u[-1] <=1 and we integrate to u<1
    # compute integrand on grid points (except maybe last)
    integrand = (eta1 - eta_grid) / (1.0 - grid_u + 1e-16)  # add tiny eps to avoid strict zero division at v=1
    # cumulative integral via composite trapezoid:
    I = np.zeros_like(eta_grid)
    # trapezoidal integrate integrand from 0 to u_i
    # compute widths:
    du = np.diff(grid_u)
    # integral at first point is zero
    for i in range(1, grid_u.size):
        # trapezoid on [u_{i-1}, u_i]
        I[i] = I[i-1] + 0.5 * (integrand[i-1] + integrand[i]) * du[i-1]

    Zhat = (eta_grid - I) / np.sqrt(mu_hat)
    return Zhat

# ---------------------------
# 4) build vector Zhat_i (step (iv) from the paper)
# ---------------------------
def increments_from_Zhat(Zhat_grid, grid_u, n, tau=1.0):
    """
    Sample Zhat_grid at points u_i = i * tau / n for i = 0..n, then build
    Zhat_i := sqrt(n/tau) * ( Zhat(u_i) - Zhat(u_{i-1}) ), i=1..n
    Returns a vector of n values.
    """
    # target evaluation points:
    us = np.linspace(0.0, tau, n+1)
    # interpolate Zhat_grid at these us (grid_u may be finer)
    Z_interp = np.interp(us, grid_u, Zhat_grid)
    diffs = np.diff(Z_interp)  # length n
    Zi = np.sqrt(n / tau) * diffs
    return Zi
