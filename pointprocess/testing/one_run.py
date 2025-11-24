import numpy as np
from pointprocess.estimation.mle import fit_hawkes
from pointprocess.testing.increments import naive_increments, khmaladze_increments
from pointprocess.testing.gof import ks_test, cvm_test, ad_test_known_normal

def one_run(events, T, H0="exp", method="khmaladze",
            alpha_level=0.05, tau=1.0, n_for_test=None, grid_size=None):
    
    events = np.asarray(events, float)

    # 1. MLE
    estimated_params = fit_hawkes(events, T, H0=H0).params_dict

    # 2. Resolution
    if n_for_test is None:
        n_for_test = int(np.ceil(np.sqrt(T) / 4.0))
    if grid_size is None:
        grid_size = max(2000, 20 * n_for_test)

    # 3. Increments
    if method == "naive":
        x = naive_increments(events, estimated_params,
                             T, n_for_test, tau, grid_size, H0)
    elif method == "khmaladze":
        x = khmaladze_increments(events, estimated_params,
                                 T, n_for_test, tau, grid_size, H0)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # 4. Tests
    ks_reject  = ks_test(x, alpha_level)
    cvm_reject = cvm_test(x, alpha_level)
    ad_reject  = ad_test_known_normal(x, alpha_level)

    return ks_reject, ad_reject, cvm_reject
