from pointprocess.utils.io import result_table, analyze_table

method = "naive"
duration = "1h"
H0 = "multiexp_fixed_betas"

path = f"results/{method}_{duration}.json"

table = result_table(path, "beta0")

analyze_table(table)



