import time
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import save_params_json, load_real_data
import numpy as np

def monte_carlo_real(dates, intervals, method, H0, alpha_level=0.05, J=None,
                    json_path="results/results_real.json"):
    print(f"Starting Monte Carlo : Method={method} | H0={H0} | Alpha={alpha_level}")
      
    for date in dates:
        for start, end in intervals:
            print(f"{date} {start}-{end}")
            t0 = time.perf_counter()
            start_ = date + " " + start
            end_ = date + " " + end
            events_real, T = load_real_data(start=start_, end=end_, path=f"data/{date}.csv")

            ks_reject, ad_reject, cvm_reject, estimated_params, x = one_run(
                events=events_real,
                T=T,
                H0=H0,
                method=method,
                alpha_level=alpha_level, J=J)
            t1 = time.perf_counter()
            T = estimated_params["T"]
            if H0 == "multiexp_fixed_betas":
                branching_ratios = estimated_params["alphas"]/estimated_params["betas"]
            else:
                branching_ratios = np.array([estimated_params["alpha"]/estimated_params["beta"]])
            result = {
                "date": date,
                "interval": (start, end),
                "H0": H0,
                "method": method,
                "alpha_level": alpha_level,
                "KS": ks_reject,
                "CvM": cvm_reject,
                "AD": ad_reject,
                "estimated_params": estimated_params,
                "branching_ratios": branching_ratios,
                "time_seconds": t1 - t0,
            }
            key = f"{date}_{start}-{end}"
            # Save to json
            save_params_json(key, result, json_path=json_path)
            
    

