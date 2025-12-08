# pointprocess/testing/one_run.py
import numpy as np
from pointprocess.estimation.mle import fit_hawkes
from pointprocess.testing.increments import build_increments
from pointprocess.testing.gof_tests import ks_test, cvm_test, ad_test
from pointprocess.testing.transformations import random_time_change

def one_run(events, T, H0, method,
            alpha_level, tau=1.0, n_for_test=None, grid_size=None):
    
    events = np.asarray(events, float)

    # 1. MLE
    estimated_params = fit_hawkes(events, T, H0=H0).params_dict
    if n_for_test is None:
        n_for_test = int(np.ceil(np.sqrt(T) / 4.0))
    if grid_size is None:
        grid_size = max(2000, 20 * n_for_test)

    # 2. Goodness-of-fit test
    if method == "naive_rtc":
        x = random_time_change(events, estimated_params, T, H0)
        ks_reject  = ks_test(x, alpha_level, "exp")
        cvm_reject = cvm_test(x, alpha_level, "exp")
        ad_reject  = ad_test(x, alpha_level, "exp")
    elif method == "naive":
        x = build_increments(events, estimated_params,
                                T, n_for_test, tau, grid_size, H0, "naive")
        ks_reject  = ks_test(x, alpha_level, "normal")
        cvm_reject = cvm_test(x, alpha_level, "normal")
        ad_reject  = ad_test(x, alpha_level, "normal")
    elif method == "khmaladze":
        x = build_increments(events, estimated_params,
                                T, n_for_test, tau, grid_size, H0, "khmaladze")
        ks_reject  = ks_test(x, alpha_level, "normal")
        cvm_reject = cvm_test(x, alpha_level, "normal")
        ad_reject  = ad_test(x, alpha_level, "normal")

    return ks_reject, ad_reject, cvm_reject, estimated_params
