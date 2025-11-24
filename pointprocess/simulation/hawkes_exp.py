from .base import PointProcess
from .poisson import PoissonHomogeneous
import numpy as np
from collections import deque

class HawkesExp(PointProcess):
    def __init__(self, params):
        self.params = params
        self.events = []
        self.T = params["T"]
        self.mu = params["mu"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate_cluster()
        
    def simulate_thinning(self):    # Uses a thinning algorithm
        t = 0
        events =  []
        lambda_t = self.mu
        M = self.mu
        G = 0
        while True:
            w = np.random.exponential(1/M)
            t += w
            if t > self.T:
                break
            D = np.random.uniform(0, 1)
            G_t = G * np.exp(-self.beta * w)
            lambda_t = self.mu + self.alpha * G_t
            if D <= lambda_t / M:
                events.append(t)
                M = lambda_t + self.alpha
                G = G_t + 1
            else:
                M = lambda_t
                G = G_t
        self.events = events
    
    def simulate_cluster(self):  # Uses cluster representation
        n = self.alpha / self.beta
        poisson_h = PoissonHomogeneous({"T": self.T, "lambda": self.mu})
        frontier = deque(poisson_h.events)
        while frontier:
            t_p = frontier.popleft()
            self.events.append(t_p)
            K = np.random.poisson(n)
            for _ in range(K):
                w = np.random.exponential(1/self.beta)
                t_c = t_p + w
                if t_c < self.T:
                    frontier.append(t_c)
        self.events.sort()

    @staticmethod
    def _intensity_on_grid(times, params, events):
        mu = params["mu"]
        alpha = params["alpha"]
        beta = params["beta"]
        lam = np.empty_like(times, dtype=float)
        G = 0.0
        t_last = 0.0
        k = 0
        n = len(events)
        for i, t in enumerate(times):
            while k < n and events[k] <= t:
                dt = events[k] - t_last
                G *= np.exp(-beta * dt)
                G += 1.0
                t_last = events[k]
                k += 1
            dt = t - t_last
            lam[i] = mu + alpha * (G * np.exp(-beta * dt))
        return lam
   