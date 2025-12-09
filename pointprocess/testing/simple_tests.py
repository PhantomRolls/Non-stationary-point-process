import numpy as np
from pointprocess.testing.increments import simple_compensator_test
from scipy import stats as scipy_stats


def simple_exp_test(events, T, params_dict, H0="exp", alpha_level=0.05):
    """
    Simple exponential test: Test if inter-arrival times in compensated time
    follow Exp(1) distribution (time-rescaling theorem).
    
    This is a classical goodness-of-fit test: if the model is correct,
    the compensated inter-arrival times Λ(t_i) - Λ(t_{i-1}) should be i.i.d. Exp(1).
    
    Args:
        events: Event times array
        T: Time horizon
        params_dict: Dictionary of parameters
        H0: Model type ("exp", "pl", "multiexp", "multiexp-fixed-betas")
        alpha_level: Significance level (default 0.05)
    
    Returns:
        (ks_reject, ad_reject, cvm_reject): Boolean tuple indicating test rejections
            True = reject H0 = model does NOT fit
            False = accept H0 = model fits
    """
    events = np.asarray(events, float)
    
    # Get compensated inter-arrival times
    compensated_diffs = simple_compensator_test(events, params_dict, T, H0)
    
    # Test against Exp(1) using KS test
    ks_stat, ks_pvalue = scipy_stats.kstest(compensated_diffs, 'expon', args=(0, 1))
    ks_reject = ks_pvalue < alpha_level
    
    # Test using Anderson-Darling for exponential
    ad_result = scipy_stats.anderson(compensated_diffs, dist='expon')
    # ad_result.critical_values correspond to [15%, 10%, 5%, 2.5%, 1%]
    critical_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
    idx = critical_idx.get(alpha_level, 2)  # default to 5%
    ad_reject = ad_result.statistic > ad_result.critical_values[idx]
    
    # Test using Cramér-von Mises for exponential
    cvm_result = scipy_stats.cramervonmises(compensated_diffs, 'expon', args=(0, 1))
    cvm_reject = cvm_result.pvalue < alpha_level
    
    return ks_reject, ad_reject, cvm_reject
