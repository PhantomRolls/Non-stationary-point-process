import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pointprocess.estimation.mle import fit_hawkes
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp

# Charger les données
df = pd.read_csv("data/bdfh_snapshots_FR0000130809_20170117_5_True.csv")

# Conversion du timestamp
df['ets'] = pd.to_datetime(df['ets'], format='%Y%m%d:%H:%M:%S.%f')

# Filtrage : ne garder que les trades (T)
df_trades = df.loc[df['etype'] == 'T'].copy()
df_trades = df_trades.sort_values('ets')

# Ajout du processus de comptage
df_trades.loc[:, 'N'] = range(1, len(df_trades)+1)

# Fenêtre 10h–11h
start = pd.Timestamp("2017-01-17 10:00:00")
end   = pd.Timestamp("2017-01-17 10:10:00")

df_zoom = df_trades.loc[(df_trades['ets'] >= start) & 
                        (df_trades['ets'] <= end)].copy()   # <--- TRÈS IMPORTANT

# Création du timestamp numérique en secondes
df_zoom.loc[:, 't'] = df_zoom['ets']

# Référence : 10h00 du même jour
t0 = start   # c’est déjà un Timestamp utilisable

df_zoom.loc[:, 't_sec'] = (df_zoom['t'] - t0).dt.total_seconds()

# Tri final
df_zoom = df_zoom.sort_values('t_sec')
events_real = df_zoom['t_sec'].values

T = 600

estimated_params = fit_hawkes(
    events=df_zoom['t_sec'].values,
    T=T,
    H0="exp"
).x

print("Estimated parameters (mu, alpha, beta):", estimated_params)

plt.figure(figsize=(12,5)) 
plt.plot(events_real, df_zoom['N']-df_zoom['N'].iloc[0], drawstyle='steps-post') 
plt.title("Processus de comptage") 
plt.xlabel("Temps") 
plt.ylabel("Nombre cumulé de trades") 
plt.grid(True) 
plt.show()

hawkes = HawkesExp(params={"T": T, "mu": estimated_params[0], "alpha": estimated_params[1], "beta": estimated_params[0]})
hawkes.plot()
events_sim = hawkes.events
print(len(events_real), "événements dans la fenêtre 10h–10h10")
print(len(events_sim), "événements simulés avec les paramètres estimés")

dt_real = np.diff(np.sort(events_real))
print("dt_real min :", dt_real.min())
print("dt_real moyen :", dt_real.mean())
print("dt_real médian :", np.median(dt_real))
dt_sim = np.diff(np.sort(events_sim))
print("dt_sim min :", dt_sim.min())
print("dt_sim moyen :", dt_sim.mean())
print("dt_sim médian :", np.median(dt_sim))
