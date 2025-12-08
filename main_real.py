import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pointprocess.estimation.mle import fit_hawkes
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import load_real_data, plot_interarrival_distribution, plot_two_counting_processes


start = "2017-01-17 11:00:00"
end = "2017-01-17 12:00:00"
path = "data/bdfh_snapshots_FR0000130809_20170117_5_True.csv"

events_real, T = load_real_data(start=start, end=end, path=path)
print(len(events_real), "événements réels entre", start, "et", end, "sur une période de", T, "secondes.")
plot_interarrival_distribution(events_real, bins=100)

ks_reject, ad_reject, cvm_reject, estimated_params = one_run(
    events=events_real,
    T=T,
    H0="multiexp_fixed_betas",
    method="khmaladze",
    alpha_level=0.05)

print("estimated_params:", estimated_params)


print("Test de Kolmogorov-Smirnov rejeté :", ks_reject)
print("Test d'Anderson-Darling rejeté :", ad_reject)
print("Test de Cramér-von Mises rejeté :", cvm_reject)

hawkes = HawkesMultiExp(params=estimated_params)
events_sim = hawkes.events
hawkes.plot()
plot_two_counting_processes(events_real, events_sim, T)