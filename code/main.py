import numpy as np
import time
import Hawkes as hk
from monte_carlo_seq import monte_carlo_seq
from import_data import simulate_data
from simulator import HawkesExp, HawkesPL

# -----------------------------
# Simulation parameters
# -----------------------------
true_params = {'mu': 0.5, 'alpha': 1, 'beta': 2.0}
itv = [0, 5000]                 # Observation interval [0, T]
M = 500                         # Number of Monte Carlo replications
alpha_level = 0.2               # Significance level (0.01, 0.05 or 0.2)
underlying_process = HawkesExp  # Choose the true underlying process that generates the data (between HawkesExp and HawkesPL)
method = "naive"                # Choose the test: "naive" or "khmaladze"
H0 = "pl"                       # Null hypothesis: "exp" or "pl"

# For a potential loop
alpha_levels = [0.01, 0.05]
underlying_processes = [HawkesExp, HawkesPL]
methods = ["naive", "khmaladze"] 


for alpha_level in alpha_levels:
    for underlying_process in underlying_processes:

        print(f"=== Alpha level: {alpha_level}, Underlying process: {underlying_process.__name__}, H0: {H0} ===")
        seed0 = 1234
        file_data = "data/FR0000120271_20220103_open.csv"
        sim_data = simulate_data(file_data)

        # Parameters for the Khmaladze transformation
        tau = 1.0                 # Use the entire interval [0, 1]
        n_for_test = None         # If None → code sets n = ceil(sqrt(T)/4)
        grid_size = None          # If None → grid = max(2000, 20 * n_for_test)

        # -----------------------------
        # Monte Carlo using naive transformation
        # -----------------------------
        print(f"=== {method} transformation ===")

        result_naive = monte_carlo_seq(
            true_params=true_params,
            itv=itv,
            M=M,
            
            underlying_process=underlying_process,
            alpha_level=alpha_level,
            H0 = H0,
            method=method, 
            
            sim_data=sim_data,
            seed0=seed0,
            data="simulate",
            n_sub=None,           # Use all events
            sub_method="first_n", # Sub-sampling rule (used only for naive method)
        )

        print("Final results (naive):", result_naive)
