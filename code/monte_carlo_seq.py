import numpy as np
import csv
import os
import time
from one_run_exp import one_run


def monte_carlo_seq(M, events_generator, T,
                    H0, method, alpha_level,
                    generator_name,
                    csv_path="results/results.csv"):

    ks_rej = ad_rej = cvm_rej = 0

    t0 = time.perf_counter()

    for m in range(M):

        events = events_generator()   # simulate events

        ks_r, ad_r, cvm_r = one_run(
            events=events,
            T=T,
            H0=H0,
            method=method,
            alpha_level=alpha_level
        )

        ks_rej += ks_r
        ad_rej += ad_r
        cvm_rej += cvm_r

        print(f"  --> completed {m+1}/{M}")

    t1 = time.perf_counter()

    result = {
        "generator": generator_name,
        "M": M,
        "method": method,
        "alpha_level": alpha_level,
        "KS": ks_rej,
        "CvM": cvm_rej,
        "AD": ad_rej,
        "time_seconds": t1 - t0
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Save to CSV
    fieldnames = [
        "generator",
        "method",
        "alpha_level",
        "M",
        "KS",
        "CvM",
        "AD",
        "time_seconds",
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return result
