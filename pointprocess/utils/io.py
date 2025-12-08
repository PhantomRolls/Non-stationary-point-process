import yaml
import os
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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

def plot_interarrival_distribution(events_real, bins=50, density=False, logx=True):
    """
    Trace la distribution des temps d'inter-arrivées d'un processus de comptage.

    Parameters
    ----------
    events_real : array-like
        Timestamps triés des évènements.
    bins : int
        Nombre de bins.
    density : bool
        Si True, normalise l'histogramme.
    logx : bool
        Si True, utilise une échelle logarithmique sur x et des bins log-spaced.
    """

    events = np.asarray(events_real)

    if len(events) < 2:
        raise ValueError("Il faut au moins deux événements pour calculer les inter-arrivées.")

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
    plt.figure(figsize=(8, 4))
    plt.hist(inter_pos, bins=used_bins, density=density, edgecolor='k', alpha=0.7)

    if logx:
        plt.xscale("log")

    plt.xlabel("Temps inter-arrivées" + (" (log)" if logx else ""))
    plt.ylabel("Densité" if density else "Fréquence")
    plt.title("Distribution des temps d'inter-arrivées")
    plt.grid(True, alpha=0.3)
    plt.show()


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
