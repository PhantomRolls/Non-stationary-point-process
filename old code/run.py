from simulator import HawkesExp, HawkesPL, PoissonHomogeneous, PoissonInhomogeneous, HawkesMultiExp
from mle import fit_hawkes
import yaml
import numpy as np
import pandas as pd
import time

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()



start = time.time()             


process = PoissonInhomogeneous
params = config[process.__name__]
params["T"] = 50
hawkes = process(params)

events = hawkes.events
hawkes.plot()

end = time.time() 
print("Execution time:", end - start, "seconds")