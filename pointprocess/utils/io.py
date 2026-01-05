import yaml
import os
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import json

def load_config(path="config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

def load_real_data(start, end, path):
    T = (datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).total_seconds()
    df = pd.read_csv(path)
    df['ets'] = pd.to_datetime(df['ets'], format='%Y%m%d:%H:%M:%S.%f')
    df_trades = df.loc[df['etype'] == 'T'].copy()
    df_trades = df_trades.sort_values('ets')
    df_trades.loc[:, 'N'] = range(1, len(df_trades)+1)
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)
    df_zoom = df_trades.loc[(df_trades['ets'] >= start) & 
                            (df_trades['ets'] <= end)].copy()
    df_zoom.loc[:, 't'] = df_zoom['ets']
    t0 = start 
    df_zoom.loc[:, 't_sec'] = (df_zoom['t'] - t0).dt.total_seconds()
    df_zoom = df_zoom.sort_values('t_sec')
    events_real = df_zoom['t_sec'].values
    return events_real, T

def save_results_to_csv(result, csv_path):
    """
    Append one result dictionary to a CSV file,
    creating the directory and header if needed.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

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


def save_params_json(key, params, json_path):
    def to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = params
    data = to_json_serializable(data)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        

def plot_two_counting_processes(events1, events2, T=None):
    events1 = np.sort(np.asarray(events1))
    events2 = np.sort(np.asarray(events2))

    if T is None:
        T = max(events1.max(), events2.max())

    # Construire N(t) pour chaque processus
    times = np.linspace(0, T, 1000)

    N1 = np.searchsorted(events1, times, side="right")
    N2 = np.searchsorted(events2, times, side="right")

    plt.figure(figsize=(12, 4))
    plt.plot(times, N1, label="estimated", color="blue")
    plt.plot(times, N2, label="real", color="red")

    plt.xlabel("Time")
    plt.ylabel("N(t)")
    plt.title("Counting processes N1(t) and N2(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

def plot_interarrival_distribution(events_real, bins=50, density=False, logx=True, ax=None):
    events = np.asarray(events_real)

    inter = np.diff(events)

    # Empêche les valeurs <= 0 pour le log
    inter_pos = inter[inter > 0]

    if logx:
        # Bins log-spaced
        min_i = inter_pos.min()
        max_i = inter_pos.max()
        log_bins = np.logspace(np.log10(min_i), np.log10(max_i), bins)
        used_bins = log_bins
    else:
        used_bins = bins

    # Tracé
    if ax is None:
        ax = plt.gca()
    ax.hist(inter_pos, bins=used_bins, density=density, edgecolor='k', alpha=0.7)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Inter-Arrival Time" + (" (log)" if logx else ""))
    ax.set_ylabel("Density" if density else "Frequency")
    ax.set_title("Distribution of Inter-Arrival Times")
    ax.grid(True, alpha=0.3)


def plot_bic(scores, J_max, criterion, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.plot(range(1, J_max + 1), scores, marker="o")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel(criterion.upper())
        ax.set_title("Selection of the Number of GMM Components")


from scipy.stats import lognorm

def plot_gmm_interarrival_counts(
    events_real,
    gmm,
    bins,
    ax=None,
    plot_components=True,
    plot_total=True,
):
    events = np.asarray(events_real)
    inter = np.diff(events)
    inter = inter[inter > 0]

    if ax is None:
        ax = plt.gca()

    # Recréer EXACTEMENT les mêmes bins que l'histogramme
    bin_edges = bins if np.ndim(bins) > 0 else np.histogram_bin_edges(inter, bins=bins)

    N = len(inter)
    counts_total = np.zeros(len(bin_edges) - 1)

    # Comptes attendus par bin
    for j in range(gmm.n_components):
        mu = gmm.means_[j, 0]
        sigma = np.sqrt(gmm.covariances_[j, 0, 0])
        w = gmm.weights_[j]

        # Probabilité par bin
        cdf = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
        probs = np.diff(cdf)

        counts = N * w * probs
        counts_total += counts

        if plot_components:
            ax.step(
                bin_edges[:-1],
                counts,
                where="post",
                lw=2,
                label=f"Comp {j+1} | τ={np.exp(mu):.2e}"
            )

    if plot_total:
        ax.step(
            bin_edges[:-1],
            counts_total,
            where="post",
            lw=2.5,
            color="k",
            linestyle="--",
            label="GMM total"
        )



from scipy.stats import lognorm

def annotate_gmm_weights(
    events_real,
    gmm,
    bins,
    ax,
    fontsize=10,
    alpha=0.75,
    y_mult=1.10,          # combien au-dessus du pic
    stack_mult=1.18,      # empilement si collision (même bin)
    min_weight=0.0        # ex: 0.03 pour ignorer petits poids
):
    events = np.asarray(events_real)
    inter = np.diff(events)
    inter = inter[inter > 0]
    N = len(inter)

    bin_edges = np.asarray(bins, dtype=float)
    if bin_edges.ndim == 0:
        bin_edges = np.histogram_bin_edges(inter, bins=bins)

    used_bins = {}  # k -> combien de labels déjà posés sur ce bin

    for j in range(gmm.n_components):
        w = float(gmm.weights_[j])
        if w < min_weight:
            continue

        mu = float(gmm.means_[j, 0])
        sigma = float(np.sqrt(gmm.covariances_[j, 0, 0]))

        # Counts attendus par bin pour CETTE composante (exactement comme le step)
        cdf = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
        probs = np.diff(cdf)
        counts = N * w * probs

        if not np.any(np.isfinite(counts)) or counts.max() <= 0:
            continue

        kmax = int(np.nanargmax(counts))

        # x au centre "log" du bin (mieux en échelle log)
        x_left, x_right = bin_edges[kmax], bin_edges[kmax + 1]
        x_peak = np.sqrt(x_left * x_right)  # moyenne géométrique
        y_peak = float(counts[kmax])

        # si plusieurs labels tombent sur le même bin, on empile légèrement
        n_here = used_bins.get(kmax, 0)
        used_bins[kmax] = n_here + 1
        y = y_peak * (y_mult * (stack_mult ** n_here))

        ax.text(
            x_peak,
            y,
            f"$w_{j+1}={w:.2f}$",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            alpha=alpha,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.45)
        )

def qq_plot(x):
    import scipy.stats as stats
    n = len(x)
    q = stats.expon.ppf((np.arange(1, n+1) - 0.5) / n)
    x_exp = np.random.exponential(scale=1.0, size=len(x))
    plt.scatter(q, np.sort(x), s=12, alpha=0.6, label="x")
    plt.scatter(q, np.sort(x_exp), s=12, alpha=0.6, label="Exp(1)")

    # VRAIE droite de référence
    plt.plot(q, q, "r-", lw=2, label="y = x (Exp(1))")

    plt.xlabel("Quantiles théoriques Exp(1)")
    plt.ylabel("Quantiles empiriques")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    

def result_table(path, value, log=True):
    # Charger le fichier JSON
    with open(path, "r") as f:
        data = json.load(f)
    records = []
    for key, v in data.items():
        date = v["date"]
        start, end = v["interval"]
        hour = f"{start}-{end}"
        if value == "tests":
            val = int(v["KS"] or v["CvM"] or v["AD"])
        elif value == "J":
            params = v.get("estimated_params", {})
            val = params["J"]
        if value == "branching_ratio":
            branching_ratios = v.get("branching_ratios", {})
            val = sum(branching_ratios)
        if value == "branching_ratios":
            branching_ratios = v.get("branching_ratios", [])
            val = np.array(branching_ratios).round(2)
        elif value == "beta0":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][0]),2)
            else:
                val = round(params["betas"][0], 2)
        elif value == "beta1":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][1]),2)
            else:
                val = round(params["betas"][1], 2)
        elif value == "beta2":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][2]),2)
            else:
                val = round(params["betas"][2], 2)
        records.append({
            "date": date,
            "hour": hour,
            "values": val
        })
    df = pd.DataFrame(records)
    table = (
        df
        .pivot(index="hour", columns="date", values="values")
        .sort_index()
    )
    table.attrs["name"] = f"{value}"
    return table

def analyze_table(df):    
    if df.attrs["name"] == "beta0" or df.attrs["name"] == "beta1" or df.attrs["name"] == "beta2":
        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        print(df)
        print("Mean :", df.stack().mean(), " | ", np.exp(df.stack().mean()),np.exp(df.stack()).mean())
        df.mean(axis=0).plot(ax=axes[0])
        axes[0].set_title("Daily Average Log(Beta)")
        axes[0].set_ylabel("Beta")
        axes[0].set_xlabel("Date")

        df.T.plot(ax=axes[1], legend=False)
        axes[1].set_title("Hourly Log(Beta) Values")
        axes[1].set_ylabel("Beta")
        axes[1].set_xlabel("Date")
        
        df.T.boxplot(ax=axes[2])
        axes[2].set_title("Hourly Log(Beta) Dispersion")
        axes[2].set_ylabel("Beta")

        labels = [str(h).split(":")[0] for h in df.index]

        axes[2].set_xticks(range(1, len(labels) + 1))
        axes[2].set_xticklabels(labels, rotation=90)


        plt.tight_layout()
        plt.show()
    
    elif df.attrs["name"] == "tests":
        print(df)
        n_rejections = df.values.sum()
        print(f"Number of rejetctions : {n_rejections} ({round(n_rejections/df.size*100,1)} %)")

        from matplotlib.patches import Patch
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            df.values,
            cmap='gray_r',
            aspect='auto',
            extent=[-0.5, len(df.columns)-0.5, len(df.index)-0.5, -0.5]
        )
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right')

        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)

        ax.grid(
            which='minor',
            color='lightgray',
            linestyle='-',
            linewidth=0.7
        )

        ax.tick_params(which='minor', bottom=False, left=False)

        # Labels
        ax.set_xlabel("Date")
        ax.set_ylabel("Hour")
        ax.set_title("Accepted / Rejected Schedule")

        # Légende
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='0 : Accepted'),
            Patch(facecolor='black', edgecolor='black', label='1 : Rejected')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    elif df.attrs["name"] == "J" or df.attrs["name"] == "branching_ratio" or df.attrs["name"] == "branching_ratios":
        print(df)
        


