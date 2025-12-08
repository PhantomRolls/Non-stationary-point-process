from .base import PointProcess
from .poisson import PoissonHomogeneous
import numpy as np
from collections import deque

class HawkesMultiExp(PointProcess):
    def __init__(self, params):
        self.events = []
        self.params = params
        self.T = params["T"]
        self.mu = params["mu"]
        self.alphas = np.array(params["alphas"])
        self.betas = np.array(params["betas"]) 
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate_cluster() 

    def simulate_cluster(self):
        J = len(self.alphas)
        branching_ratios = self.alphas / self.betas   # α_j / β_j pour chaque j
        poisson_h = PoissonHomogeneous({"T": self.T, "lambda": self.mu})
        frontier = deque(poisson_h.events)

        while frontier:
            t_p = frontier.popleft()
            self.events.append(t_p)

            for j in range(J):
                # nombre d'enfants pour le kernel j
                K_j = np.random.poisson(branching_ratios[j])
                for _ in range(K_j):
                    # temps d'attente exponentiel ~ Exp(beta_j)
                    w = np.random.exponential(1 / self.betas[j])
                    t_c = t_p + w
                    if t_c < self.T:
                        frontier.append(t_c)

        self.events.sort()

    @staticmethod
    def _intensity_on_grid(times, params, events):
        mu = params["mu"]
        alphas = np.array(params["alphas"])
        betas  = np.array(params["betas"])
        J = len(alphas)

        times = np.asarray(times)
        events = np.asarray(events)

        lam = np.zeros_like(times, dtype=float)
        lam[:] = mu

        # intensité excitée pour chaque composante j
        exc = np.zeros(J)

        # pointeur pour parcourir les événements
        i = 0
        n_events = len(events)

        for k in range(1, len(times)):
            dt = times[k] - times[k-1]

            # décroissance exponentielle entre t[k-1] et t[k]
            exc *= np.exp(-betas * dt)

            # ajouter tous les événements qui tombent dans (t[k-1], t[k]]
            while i < n_events and events[i] <= times[k]:
                for j in range(J):
                    exc[j] += alphas[j]   # un "saut" α_j pour chaque événement
                i += 1

            lam[k] += exc.sum()

        return lam

