import numpy as np
from pointprocess.montecarlo.monte_carlo import monte_carlo_simulation
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp
import ctypes
ctypes.windll.kernel32.SetThreadExecutionState(0x8000000 | 0x00000001 | 0x00000002) # Prevent sleep mode on Windows

M = 50
alpha_levels = [0.2]
process_generators = [HawkesExp]
methods = ["naive", "khmaladze"]
H0 = "exp"
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

