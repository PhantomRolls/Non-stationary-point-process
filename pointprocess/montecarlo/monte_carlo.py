import numpy as np
import time
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import load_config, save_results_to_csv


           
def monte_carlo_simulation(M, process_generator,
                    H0, method, alpha_level, config_path="config.yaml",
                    csv_path="results.csv"):
    print(f"Starting Monte Carlo : M={M} | Process={process_generator.__name__} | Method={method} | H0={H0} | Alpha={alpha_level}")
    
        
    print(f"Loading parameters from config: {config_path}")
    config = load_config(config_path)

    process_params = config[process_generator.__name__]
    T = process_params["T"]
    ks_rej = ad_rej = cvm_rej = 0

    t0 = time.perf_counter()
    all_betas = []
    for m in range(M):

        events = process_generator(process_params).events

        ks_r, ad_r, cvm_r, estimated_params, x = one_run(
            events=events,
            T=T,
            H0=H0,
            method=method,
            alpha_level=alpha_level
        )

        ks_rej += ks_r
        ad_rej += ad_r
        cvm_rej += cvm_r
        
        betas = estimated_params["betas"]
        all_betas.append(betas)
        
        if (m + 1) % 10 == 0:
            print(f"{m+1}/{M}")

    medians = np.median(all_betas, axis=0)
    print(medians)
    t1 = time.perf_counter()

    result = {
        "generator": process_generator.__name__,
        "M": M,
        "method": method,
        "alpha_level": alpha_level,
        "KS": ks_rej,
        "CvM": cvm_rej,
        "AD": ad_rej,
        "time_seconds": t1 - t0,
    }

    # Save to CSV
    save_results_to_csv(result, csv_path)
    print(f"{m+1}/{M}")
    return result




    
