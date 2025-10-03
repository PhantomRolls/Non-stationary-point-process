from simulator import Hawkes, PoissonHomogeneous, PoissonInhomogeneous
from mle import fit_hawkes
from test import simulation, plot_ccdf
import numpy as np
import pandas as pd
import time

T = 100
true_params = {"mu": .5, "alpha": 1, "beta": 2.0}
n_simulations = 500

hawkes = Hawkes(T, true_params)
events = hawkes.events
hawkes.plot()
# res = fit_hawkes(events, T)
# print("Estimated parameters:", res.x)

# t0 = time.perf_counter()
# df = simulation(n_simulations, T, true_params, alpha=0.20, plot_qq=True)
# dt = time.perf_counter() - t0
# print(f"Durée: {dt:.2f} s")

# plot_ccdf(true_params, T)
