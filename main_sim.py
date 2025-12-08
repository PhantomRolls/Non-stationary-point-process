import numpy as np
from pointprocess.montecarlo.monte_carlo import monte_carlo_simulation
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp
from pointprocess.estimation.mle import fit_hawkes
import ctypes
ctypes.windll.kernel32.SetThreadExecutionState(0x8000000 | 0x00000001 | 0x00000002)

M = 500
alpha_levels = [0.01, 0.05, 0.2]
process_generators = [HawkesMultiExp, HawkesPL, HawkesExp]
methods = ["naive_rtc"]
H0 = "multiexp"
csv_path = "results/results.csv"

for process_generator in process_generators:
    for method in methods:
        for alpha_level in alpha_levels:        
            result = monte_carlo_simulation(
                M=M,
                process_generator=process_generator,
                H0=H0,
                method=method,
                alpha_level=alpha_level,
                csv_path=csv_path,
                config_path="config.yaml",
            )
            
            
# hawkes = HawkesExp({
#     "mu": 0.5,
#     "alpha": 1,
#     "beta": 1.05,
#     "T": 50
# })
# events = hawkes.events
# hawkes.plot()

