from simulator import HawkesExp, HawkesPL, PoissonHomogeneous, PoissonInhomogeneous
from mle import fit_hawkes
from test import simulation, plot_ccdf
import numpy as np
import pandas as pd
import time

T = 5000
true_params = {"mu": .5, "alpha": 1, "beta": 2.0}
n_simulations = 500


start = time.time()             
hawkes = HawkesPL(T, true_params)

events = hawkes.events
hawkes.plot()

end = time.time() 
print("Execution time:", end - start, "seconds")