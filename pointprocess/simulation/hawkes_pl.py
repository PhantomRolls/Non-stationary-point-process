from .base import PointProcess
import numpy as np
from .poisson import PoissonHomogeneous
from collections import deque

class HawkesPL(PointProcess):
    def __init__(self, params):
        self.events = []
        self.params = params
        self.T = params["T"]
        self.mu = params["mu"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate_cluster()
        
    def simulate_cluster(self):
        n = self.alpha / self.beta
        poisson_h = PoissonHomogeneous({"T": self.T, "lambda": self.mu})
        frontier = deque(poisson_h.events)
        while frontier:
            t_p = frontier.popleft()
            self.events.append(t_p)
            K = np.random.poisson(n)
            for _ in range(K):
                u = np.random.uniform(0, 1)
                w = (1 - u) ** (-1/self.beta) - 1
                t_c = t_p + w
                if t_c < self.T:
                    frontier.append(t_c)
        self.events.sort()
    
    @staticmethod
    def _intensity_on_grid(times, params, events):
        mu = params["mu"]
        alpha = params["alpha"]
        beta = params["beta"]
        times = np.asarray(times, float)
        events = np.asarray(events, float)
        lam = np.full_like(times, mu, dtype=float)
        for k, t in enumerate(times):
            past = events[events < t]
            if past.size:
                lam[k] += alpha * np.sum((1.0 + (t - past))**(-(beta+1)))
        return lam
    