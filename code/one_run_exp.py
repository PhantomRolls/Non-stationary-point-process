import numpy as np
import time
from scipy.stats import kstest, anderson, cramervonmises

import Hawkes as hk
from compensator import compensator_exp_fast, compensator_pl_exact
from clever_trasformation import (
    build_eta_on_grid,
    transform_T_eta_univariate,
    increments_from_Zhat,
)
from simulator import HawkesExp, HawkesPL
from mle import fit_hawkes



def one_run(true_params, itv, 
            
            underlying_process, alpha_level, H0, method,
            
            sim_data,  seed, data="real", n_sub=None, sub_method="equispaced", tau=1.0, n_for_test=None, grid_size=None,
            ):
    
    """
    Single Monte Carlo replication for a Hawkes process with exponential kernel,
    using either the naive time-rescaling transformation or the Khmaladze
    transformation.
    """
    # -----------------------------------------------------
    # 1) Use real or simulated data
    # -----------------------------------------------------
    if data == "real":
        events = sim_data

    elif data == "simulate":
        hawkes = underlying_process(itv[1], true_params)
        events = hawkes.events
        events = np.asarray(events, dtype=np.float64)

    else:
        raise ValueError("Invalid data option. Use 'real' or 'simulate'.")

    # -----------------------------------------------------
    # 2) Parameter estimation (With hawkes librairy)
    # -----------------------------------------------------
    #
    # est = hk.estimator()
    # est.set_kernel('exp')
    # est.set_baseline('const')
    # est.fit(events, itv)
    # p = est.parameter
    # mu_hat = float(p['mu'])
    # alpha_hat = float(p['alpha'])
    # beta_hat = float(p['beta'])
    # alpha_hat = alpha_hat * beta_hat  # Adjust alpha for the different kernel definition of the library


    # -----------------------------------------------------
    # 2) Parameter estimation (your MLE implementation)
    # -----------------------------------------------------
    T = itv[1] - itv[0]
    res = fit_hawkes(events, T, H0=H0)
    mu_hat, alpha_hat, beta_hat = res.x

    # -----------------------------------------------------
    # 3) Transformation
    # -----------------------------------------------------
    if method == "naive":
        # Compensator increments
        if H0 == "exp":
            Lambda_hat = compensator_exp_fast(mu_hat, alpha_hat, beta_hat, events)
        elif H0 == "pl":
            # Example 3-exponential approximation (can be tuned)
            Lambda_hat = compensator_pl_exact(mu_hat, alpha_hat, beta_hat, events)
        Lambda_prev0 = np.concatenate(([0.0], Lambda_hat[:-1]))
        x_full = Lambda_hat - Lambda_prev0

        # Subsampling
        if n_sub is not None and n_sub < len(x_full):
            if sub_method == "first_n":
                x = x_full[:n_sub]
            elif sub_method == "equispaced":
                idx = np.linspace(0, len(x_full) - 1, n_sub, dtype=int)
                x = x_full[idx]
            else:
                raise ValueError("Invalid subsampling method.")
        else:
            x = x_full

    elif method == "khmaladze":
        T = itv[1]

        if n_for_test is None:
            n_for_test = int(np.ceil(np.sqrt(T) / 4.0))

        if grid_size is None:
            grid_size = max(2000, 20 * n_for_test)

        grid_u = np.linspace(0.0, tau, grid_size)

        if tau < 1.0:
            grid_u_full = np.linspace(0.0, 1.0, grid_size)
        else:
            grid_u_full = grid_u.copy()

        eta_full, counts_full, Lambda_full = build_eta_on_grid(
            events, mu_hat, alpha_hat, beta_hat, T, grid_u_full
        )

        if tau < 1.0:
            mask = grid_u_full <= tau
            grid_u_tau = grid_u_full[mask]
            eta_tau = eta_full[mask]
        else:
            grid_u_tau = grid_u_full
            eta_tau = eta_full

        Zhat_tau = transform_T_eta_univariate(
            eta_tau, grid_u_tau, len(events) / T
        )

        x = increments_from_Zhat(
            Zhat_tau, grid_u_tau, n_for_test, tau=tau
        )

    else:
        raise ValueError("Invalid method. Use 'naive' or 'khmaladze'.")

    # -----------------------------------------------------
    # 4) Normality / Exponentiality tests
    # -----------------------------------------------------
    ks_dist = 'expon' if method == "naive" else 'norm'
    ks_stat, ks_p = kstest(x, ks_dist)
    ks_reject = (ks_p < alpha_level)

    if method == "naive":
        ad_res = anderson(x, dist='expon')
    else:
        ad_res = anderson(x, dist='norm')

    ad_stat = ad_res.statistic
    levels = np.array(ad_res.significance_level) / 100.0
    crit_vals = np.array(ad_res.critical_values)
    idx = np.argmin(np.abs(levels - alpha_level))
    ad_reject = (ad_stat > crit_vals[idx])

    cvm_dist = 'expon' if method == "naive" else 'norm'
    cvm_res = cramervonmises(x, cvm_dist)
    cvm_reject = (cvm_res.pvalue < alpha_level)

    return ks_reject, ad_reject, cvm_reject, True, {
        "method": method,
        "mu_hat": mu_hat,
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "n_events": len(events)
    }
