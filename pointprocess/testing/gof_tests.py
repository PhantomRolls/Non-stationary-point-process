# pointprocess/testing/gof.py
import numpy as np
from scipy.stats import kstest, cramervonmises, norm

def ks_test(x: np.ndarray, alpha: float):
    stat, p = kstest(x, 'norm')
    return p < alpha

def cvm_test(x: np.ndarray, alpha: float):
    res = cramervonmises(x, 'norm')
    return res.pvalue < alpha

def ad_statistic_known_normal(x: np.ndarray) -> float:
    """
    Anderson–Darling statistic for Normal(0,1) with KNOWN params.
    """
    x = np.sort(x)
    n = len(x)
    Fi = np.clip(norm.cdf(x), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(Fi) + np.log(1 - Fi[::-1])))

def ad_test_known_normal(x: np.ndarray, alpha: float):
    crit = {0.01: 3.75, 0.05: 2.49, 0.10: 2.14, 0.20: 1.40}[alpha]
    A2 = ad_statistic_known_normal(x)
    return A2 > crit
