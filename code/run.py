from .simulator import Hawkes, PoissonHomogeneous, PoissonInhomogeneous, SelfCorrecting, ShotNoise
from .mle import fit_hawkes, fit_self_correcting
from .test import simulation, plot_ccdf, simulation_self_correcting
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

if __name__ == "__main__":
    print("\n=== Testing Self-Correcting Point Process ===")
    T_sc = 100
    true_params_sc = {"mu": 0.5, "alpha": 0.3, "beta": 0.1}
    n_simulations_sc = 100
    
    t0 = time.perf_counter()
    df_sc = simulation_self_correcting(n_simulations_sc, T_sc, true_params_sc, alpha=0.05, plot_qq=False)
    dt = time.perf_counter() - t0
    print(f"Duration: {dt:.2f} s")
    
    print("\n=== Testing Shot Noise Process ===")
    shot_noise = ShotNoise(T, {"mu": 0.5, "alpha": 1.0, "beta": 2.0})
    print(f"Shot Noise generated {len(shot_noise.events)} events")
    shot_noise.plot()
