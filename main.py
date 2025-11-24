import numpy as np
from pointprocess.montecarlo.monte_carlo import monte_carlo_simulation
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp


M = 500
alpha_levels = [0.01]
process_generators = [HawkesMultiExp]
methods = ["khmaladze"]
H0 = "pl"
csv_path = "results/results_pl.csv"

for alpha_level in alpha_levels:
    for process_generator in process_generators:
        for method in methods:
            result = monte_carlo_simulation(
                M=M,
                process_generator=process_generator,
                H0=H0,
                method=method,
                alpha_level=alpha_level,
                csv_path=csv_path,
                config_path="config.yaml",
            )

