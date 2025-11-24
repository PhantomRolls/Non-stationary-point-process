import numpy as np
from scipy.stats import kstest, anderson, cramervonmises
from scipy.stats import norm
from clever_trasformation import (
    build_eta_on_grid,
    transform_T_eta_univariate,
    increments_from_Zhat,
)
from mle import fit_hawkes

def naive_increments(events, mu_hat, alpha_hat, beta_hat, T, n, tau, grid_size, H0):

    # Grid u in [0, tau]
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, mu_hat, alpha_hat, beta_hat, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Standardized empirical process W̃(u)
    mean_intensity_hat = len(events) / T
    Wtilde = eta_tau / np.sqrt(mean_intensity_hat)

    # Increments Z̃_i (eq. 24)
    Ztilde = increments_from_Zhat(Wtilde, grid_u, n, tau=tau)
    return Ztilde


def khmaladze_increments(events, mu_hat, alpha_hat, beta_hat, T, n, tau, grid_size, H0):

    # Grid u
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, mu_hat, alpha_hat, beta_hat, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Khmaladze transform Ŵ(u)
    mean_intensity_hat = len(events) / T
    What = transform_T_eta_univariate(eta_tau, grid_u, mean_intensity_hat)

    # Increments Ẑ_i
    Zhat = increments_from_Zhat(What, grid_u, n, tau=tau)
    return Zhat


def one_run(events, T, H0="exp", method="khmaladze",
            alpha_level=0.05, tau=1.0, n_for_test=None, grid_size=None):
    """
    Single test run (naive or Khmaladze), minimal version.
    Only depends on:
        - events
        - T
        - H0 ("exp" or "pl")
        - method ("naive" or "khmaladze")
    """

    events = np.asarray(events, float)

    # 1. Estimate Hawkes parameters
    mu_hat, alpha_hat, beta_hat = fit_hawkes(events, T, H0=H0).x

    # 2. Choose resolution
    if n_for_test is None:
        n_for_test = int(np.ceil(np.sqrt(T) / 4.0))
    if grid_size is None:
        grid_size = max(2000, 20 * n_for_test)

    # 3. Compute increments
    if method == "naive":
        x = naive_increments(events, mu_hat, alpha_hat, beta_hat,
                             T, n_for_test, tau, grid_size, H0)

    elif method == "khmaladze":
        x = khmaladze_increments(events, mu_hat, alpha_hat, beta_hat,
                                 T, n_for_test, tau, grid_size, H0)

    # 4. Normality tests
    ks_stat, ks_p = kstest(x, 'norm')
    ks_reject = (ks_p < alpha_level)

    # ad_res = anderson(x, 'norm')
    # levels = np.array(ad_res.significance_level) / 100
    # critvals = np.array(ad_res.critical_values)
    # idx = np.argmin(abs(levels - alpha_level))
    # ad_reject = (ad_res.statistic > critvals[idx])

    cvm_res = cramervonmises(x, 'norm')
    cvm_reject = (cvm_res.pvalue < alpha_level)
    
    def AD_normal_known(x):
        """
        Anderson–Darling test for normality with KNOWN mean=0 and variance=1.
        Returns:
        A2 : the AD statistic
        """
        x = np.sort(x)
        n = len(x)
        Fi = norm.cdf(x)
        # Avoid log(0)
        Fi = np.clip(Fi, 1e-12, 1 - 1e-12)
        i = np.arange(1, n+1)
        A2 = -n - np.mean((2*i - 1) * (np.log(Fi) + np.log(1 - Fi[::-1])))
        return A2

    def AD_test_normal_known(x, alpha):
        A2 = AD_normal_known(x)
        crit = {0.01: 3.75, 0.05: 2.49, 0.10: 2.14, 0.20: 1.40}[alpha]
        reject = A2 > crit
        return reject, A2, crit

    ad_reject, A2, crit = AD_test_normal_known(x, alpha_level)

    return ks_reject, ad_reject, cvm_reject
