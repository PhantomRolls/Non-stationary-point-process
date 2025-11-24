import numpy as np
from simulator import HawkesExp, HawkesPL
from monte_carlo_seq import monte_carlo_seq


# ---------------------------------------------------
# 1. Define generator of events
# ---------------------------------------------------
def make_generator(process_class, params):
    generator_name = process_class.__name__
    def generator():
        H = process_class(params)
        return H.events
    return generator, generator_name


# ---------------------------------------------------
# 2. Monte Carlo settings
# ---------------------------------------------------
M = 500
T = 5000

alpha_levels = [0.01, 0.05, 0.2]
underlying_processes = [HawkesExp, HawkesPL]
methods = ["naive", "khmaladze"]
H0 = "pl"
csv_path = "results/results_pl.csv"

for alpha_level in alpha_levels:
    for process_class in underlying_processes:
        for method in methods:

            print(f"=== α={alpha_level}, underlying={process_class.__name__}, H0={H0}, method={method} ===")

            # True parameters of the generating process
            true_params = {
                "mu": 0.5,
                "alpha": 1.0,
                "beta": 2.0,
                "T": T
            }

            # Build event generator
            events_generator, generator_name = make_generator(process_class, true_params)

            # Run Monte Carlo
            result = monte_carlo_seq(
                M=M,
                events_generator=events_generator,
                T=T,
                H0=H0,
                method=method,
                alpha_level=alpha_level,
                generator_name=generator_name,
                csv_path=csv_path
            )

            print("Result:", result)
