# pointprocess/testing/gof.py
import numpy as np
from scipy.stats import kstest, cramervonmises, norm

def ks_test(x: np.ndarray, alpha: float, dist: str):
    if dist == "normal":
        stat, p = kstest(x, 'norm')          # Normal(0,1)
    elif dist == "exp":
        stat, p = kstest(x, 'expon')         # Exp(1)
    else:
        raise ValueError("dist must be 'normal' or 'exp'")
    return p < alpha

def cvm_test(x: np.ndarray, alpha: float, dist: str):
    if dist == "normal":
        res = cramervonmises(x, 'norm')      # Normal(0,1)
    elif dist == "exp":
        res = cramervonmises(x, 'expon')     # Exp(1)
    else:
        raise ValueError("dist must be 'normal' or 'exp'")
    return res.pvalue < alpha


def ad_statistic(x: np.ndarray, dist: str) -> float:
    x = np.sort(x)
    n = len(x)

    if dist == "normal":
        Fi = norm.cdf(x)

    elif dist == "exp":
        Fi = 1 - np.exp(-x)   # CDF Exp(1)

    else:
        raise ValueError("dist must be 'normal' or 'exp'")

    Fi = np.clip(Fi, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)

    return -n - np.mean((2 * i - 1) * (np.log(Fi) + np.log(1 - Fi[::-1])))


def ad_test(x: np.ndarray, alpha: float, dist: str):
    if dist == "normal":
        crit = {0.01: 3.75, 0.05: 2.49, 0.10: 2.14, 0.20: 1.40}[alpha]

    elif dist == "exp":
        crit = {0.01: 1.957, 0.05: 1.335, 0.10: 1.072, 0.20: 0.798}[alpha]

    else:
        raise ValueError("dist must be 'normal' or 'exp'")

    A2 = ad_statistic(x, dist)
    return A2 > crit

