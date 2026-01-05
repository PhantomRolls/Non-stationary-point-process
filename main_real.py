import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp
from pointprocess.montecarlo.monte_carlo_real import monte_carlo_real
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import load_real_data, qq_plot, plot_two_counting_processes
from pointprocess.utils.io import load_config

dates = ["2017-01-20", "2017-01-23", "2017-01-24", "2017-01-25", "2017-01-26", "2017-01-27", "2017-01-30", "2017-01-31", "2017-02-01"]  
intervals_1h = [("09:00:00", "10:00:00"), ("10:00:00", "11:00:00"), ("11:00:00", "12:00:00"), ("12:00:00", "13:00:00"), ("13:00:00", "14:00:00"), ("14:00:00", "15:00:00"), ("15:00:00", "16:00:00")]
intervals_30min = [
    ("09:00:00", "09:30:00"),
    ("09:30:00", "10:00:00"),
    ("10:00:00", "10:30:00"),
    ("10:30:00", "11:00:00"),
    ("11:00:00", "11:30:00"),
    ("11:30:00", "12:00:00"),
    ("12:00:00", "12:30:00"),
    ("12:30:00", "13:00:00"),
    ("13:00:00", "13:30:00"),
    ("13:30:00", "14:00:00"),
    ("14:00:00", "14:30:00"),
    ("14:30:00", "15:00:00"),
    ("15:00:00", "15:30:00"),
    ("15:30:00", "16:00:00"),
    ("16:00:00", "16:30:00"),
    ("16:30:00", "17:00:00"),
    ("17:00:00", "17:30:00"),
]

# monte_carlo_real(dates, intervals_30min, H0="multiexp_fixed_betas", method="khmaladze", J=3)
    
 


# date = "2017-01-23"
# start = "11:00:00"
# end = "12:00:00"

# start_ = date + " " + start
# end_ = date + " " + end
# events_real, T = load_real_data(start=start_, end=end_, path=f"data/{date}.csv")

# config = load_config("config.yaml")
# process_params = config[HawkesMultiExp.__name__]
# T = process_params["T"]
# events_sim = HawkesMultiExp(process_params).events
# x_exp = np.random.exponential(scale=1.0, size=len(events_sim))


# ks_reject, ad_reject, cvm_reject, estimated_params, x = one_run(
#                 events=events_real,
#                 T=T,
#                 H0="multiexp_fixed_betas",
#                 method="naive_rtc",
#                 alpha_level=0.05, plot=False, J=3)

# # hawkes = HawkesMultiExp(estimated_params)
# # events_sim = hawkes.events
# # plot_two_counting_processes(events_sim, events_real)

# qq_plot(x)
