import pandas as pd
from tabulate import tabulate

def print_hawkes_table(csv_file="results/results_multiexp.csv"):
    df = pd.read_csv(csv_file)

    # Ordre des alpha comme dans ton papier
    alphas = [0.01, 0.05, 0.20]
    stats = ["KS", "CvM", "AD"]
    methods = [("khmaladze", "Transf."), ("naive", "Naive"), ("naive_rtc", "Naive RTC")]

    processes = sorted(df["generator"].unique())
    processes.insert(2, processes.pop(processes.index('HawkesMultiExp')))

    if "pl" in csv_file:
        processes.insert(0, processes.pop(processes.index('HawkesPL'))) # For HawkesPL in first
    if "multiexp" in csv_file:
        processes.insert(0, processes.pop(processes.index('HawkesMultiExp'))) # For HawkesMultiExp in first
    table_rows = []

    for proc in processes:
        row = [proc]

        for method_name, _ in methods:
            for stat in stats:
                vals = []
                for a in alphas:
                    sub = df[
                        (df["generator"] == proc)
                        & (df["method"] == method_name)
                        & (df["alpha_level"] == a)
                    ]
                    if not sub.empty:
                        vals.append(str(int(sub.iloc[0][stat])))
                    else:
                        vals.append("-")
                # format "v1; v2; v3" (R_0.01; R_0.05; R_0.20)
                row.append("; ".join(vals))

        table_rows.append(row)

    # En-têtes simples mais explicites
    headers = [
        "Test",
        "KS (Transf.)", "CvM (Transf.)", "AD (Transf.)",
        "KS (Naive)",  "CvM (Naive)",  "AD (Naive)",
        "KS (Naive RTC)",  "CvM (Naive RTC)",  "AD (Naive RTC)",
    ]

    print("\n=== Résumé des tests (R_0.01; R_0.05; R_0.20) ===\n")
    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="fancy_grid",   # tu peux essayer "github", "grid", etc.
            stralign="center",
        )
    )


if __name__ == "__main__":
    print_hawkes_table()