import numpy as np
from scipy.stats import kstest, kstwobign
import pandas as pd
from tabulate import tabulate
from .simulator import Hawkes, SelfCorrecting
from .mle import fit_hawkes, fit_self_correcting
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

def random_time_change_self_correcting(events, params):
    """Random time change for self-correcting point process."""
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    transformed_events = []
    
    for i, ti in enumerate(events):
        if i == 0:
            ti_prev = 0
        else:
            ti_prev = events[i-1]
        
        if abs(beta) > 1e-8:
            zi = (np.exp(mu - alpha * i) / beta) * (np.exp(beta * ti) - np.exp(beta * ti_prev))
        else:
            zi = np.exp(mu - alpha * i) * (ti - ti_prev)
        
        transformed_events.append(zi)
    
    return np.cumsum(transformed_events)

def simulation_self_correcting(n_simulations, T, true_params, alpha=0.05, plot_qq=False):
    """Goodness of fit testing for self-correcting point process."""
    n_rejections = 0
    stats = []
    for _ in tqdm(range(n_simulations), desc="Processing"):
        sc = SelfCorrecting(T, true_params)
        events = sc.events
        res = fit_self_correcting(events, T)
        params = {"mu": res.x[0], "alpha": res.x[1], "beta": res.x[2]}
        transformed_events = random_time_change_self_correcting(events, true_params)
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
        plt.title("Q-Q plot des statistiques KS (Self-Correcting)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.axis('equal')
        plt.show()
    return df

if __name__ == "__main__":
    print("Testing Self-Correcting Point Process")
    T = 100
    true_params = {"mu": 0.5, "alpha": 0.3, "beta": 0.1}
    n_simulations = 100
    
    import time
    t0 = time.perf_counter()
    df = simulation_self_correcting(n_simulations, T, true_params, alpha=0.05, plot_qq=False)
    dt = time.perf_counter() - t0
    print(f"Duration: {dt:.2f} s")
    
