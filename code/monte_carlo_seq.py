import numpy as np
import time
from one_run_exp import one_run
import os
import csv

def monte_carlo_seq(true_params, itv, M, 
                    
                    underlying_process, alpha_level, H0, method, 
                    
                    sim_data, seed0=0, data="real", n_sub=None, 
                    sub_method="equispaced", tau=1.0, n_for_test=None, grid_size=None):
    """
    Perform M Monte Carlo replications for a Hawkes process with exponential kernel,
    using either the naive time-rescaling transformation or the Khmaladze transformation.

    Parameters
    ----------
    true_params : dict
        True parameters {'mu': ..., 'alpha': ..., 'beta': ...}
    itv : list
        Observation interval [0, T]
    M : int
        Number of replications
    sim_data : any
        Data or data structure used to drive the simulation
    underlying_process : class or callable
        The underlying process to simulate (e.g. HawkesExp, HawkesPL)
    alpha_level : float
        Significance level of the test
    seed0 : int
        Initial random seed
    data : str
        "real" or "simulate" to indicate data source/usage
    method : str
        "naive" or "khmaladze"
    n_sub : int or None
        Subsampling of events (only for the naive method)
    sub_method : str
        Subsampling method: 'first_n' or 'equispaced' (only for naive)
    tau : float
        Tau parameter for the Khmaladze method
    n_for_test : int or None
        Final number of increments used for the test (Khmaladze)
    grid_size : int or None
        Grid size for continuous-time computations (Khmaladze)

    Returns
    -------
    dict
        Dictionary with summary statistics and runtime, also written to 'results.csv'.
    """

    ks_rejects, ad_rejects, cvm_rejects, successes = 0, 0, 0, 0
    fail_count = 0

    t0 = time.perf_counter()
    for m in range(M):
        seed = seed0 + m
        ks_r, ad_r, cvm_r, success, info = one_run(
            true_params, itv,  
            
            underlying_process=underlying_process, alpha_level=alpha_level, H0=H0, method=method,  
             
            sim_data=sim_data, seed=seed, data=data, n_sub=n_sub, sub_method=sub_method, tau=tau, n_for_test=n_for_test, grid_size=grid_size,     
        )

        if success:
            successes += 1
            if ks_r:
                ks_rejects += 1
            if ad_r:
                ad_rejects += 1
            if cvm_r:
                cvm_rejects += 1
        else:
            fail_count += 1
            print(f"[Replication {m}] Failed: {info}")

        # Print progress every 10 replications
        if (m + 1) % 1 == 0:
            print(f"  --> completed {m + 1}/{M} replications")

    t1 = time.perf_counter()

    ks_rate = ks_rejects  # / successes if successes > 0 else np.nan
    ad_rate = ad_rejects  # / successes if successes > 0 else np.nan
    cvm_rate = cvm_rejects  # / successes if successes > 0 else np.nan

    result_dict = {
        "M": M,
        "method": method,
        "KS": ks_rate,
        "CvM": cvm_rate,
        "AD": ad_rate,
        "alpha_level": alpha_level,
        "underlying_process": underlying_process.__name__,
        "time_seconds": t1 - t0
    }

    # Save the results to a CSV file
    # Run multiple times with differents parameters will append to the same file so you can print all results together in a nice chart
    FILENAME = "results/results.csv"
    fieldnames = [
        "underlying_process",
        "method",
        "alpha_level",
        "M",
        "KS",
        "CvM",
        "AD",
        "time_seconds",
    ]
    file_exists = os.path.isfile(FILENAME)
    with open(FILENAME, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)

    return result_dict
