import numpy as np
import time
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import load_config
from pointprocess.utils.io import save_results_to_csv
from pointprocess.utils.logging import setup_logger
logger = setup_logger(__name__)


           
def monte_carlo_simulation(M, process_generator,
                    H0, method, alpha_level, config_path,
                    csv_path="results.csv"):
    logger.error(f"Starting Monte Carlo : M={M} | Process={process_generator.__name__} | Method={method} | H0={H0} | Alpha={alpha_level}")
    
        
    logger.info(f"Loading parameters from config: {config_path}")
    config = load_config(config_path)

    process_params = config[process_generator.__name__]
    T = process_params["T"]
    ks_rej = ad_rej = cvm_rej = 0

    t0 = time.perf_counter()
    for m in range(M):

        events = process_generator(process_params).events

        ks_r, ad_r, cvm_r, _ = one_run(
            events=events,
            T=T,
            H0=H0,
            method=method,
            alpha_level=alpha_level
        )

        ks_rej += ks_r
        ad_rej += ad_r
        cvm_rej += cvm_r

        if (m + 1) % 10 == 0:
            logger.info(f"{m+1}/{M}")

    t1 = time.perf_counter()

    result = {
        "generator": process_generator.__name__,
        "M": M,
        "method": method,
        "alpha_level": alpha_level,
        "KS": ks_rej,
        "CvM": cvm_rej,
        "AD": ad_rej,
        "time_seconds": t1 - t0
    }

    # Save to CSV
    save_results_to_csv(result, csv_path)
    logger.info(f"{m+1}/{M}")
    return result
