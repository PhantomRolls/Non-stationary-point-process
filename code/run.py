from simulator import HawkesExp, HawkesPL, PoissonHomogeneous, PoissonInhomogeneous
from mle import fit_hawkes
from test import simulation, plot_ccdf
import numpy as np
import pandas as pd
import time
import Hawkes as hk


T = 500
true_params = {"mu": .5, "alpha": 1, "beta": 2.0}
n_simulations = 500


start = time.time() 
    
 
hawkes = HawkesExp(T, true_params)

events= hawkes.events

# est = hk.estimator()
# est.set_kernel('exp')
# #est.set_kernel('pow')
# est.set_baseline('const')
# est.fit(events, [0, T])
# p = est.parameter
# mu_hat = float(p['mu'])
# alpha_hat =float(p['alpha'])
# beta_hat = float(p['beta'])

res = fit_hawkes(events, T)


# print("Estimated parameters:", mu_hat, alpha_hat, beta_hat)

print("MLE parameters:", res.x)

end = time.time() 
print("Execution time:", end - start, "seconds") 
