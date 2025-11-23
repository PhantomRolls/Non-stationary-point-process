import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class PointProcess:
    def plot(self):
        if self.__class__.__name__ == "PoissonHomogeneous":
            self.lambda_values = self.lambda_
        elif self.__class__.__name__ == "PoissonInhomogeneous":
            self.lambda_values = PoissonInhomogeneous.lambda_t(self.times)
        elif self.__class__.__name__ == "HawkesExp":
            self.lambda_values = HawkesExp._intensity_on_grid(self.times, self.mu, self.alpha, self.beta, self.events)
        elif self.__class__.__name__ == "HawkesPL":
            self.lambda_values = HawkesPL._intensity_on_grid(self.times, self.mu, self.alpha, self.beta, self.events)
        elif self.__class__.__name__ == "HawkesMultiExp":
            self.lambda_values = HawkesMultiExp._intensity_on_grid(self.times, self.mu, self.alphas, self.betas, self.events)
        cumul = np.arange(1, len(self.events) + 1)
        fig, ax1 = plt.subplots(figsize=(9, 4))
        if len(self.events):
            (line1,) = ax1.step(self.events, cumul, where='post',
                                label="N(t)", color='navy')
        else:
            (line1,) = ax1.plot([], [], label="N(t)", color='navy')
        ax1.set_xlabel("t")
        ax1.set_ylabel("Comptage cumulatif N(t)")
        ax1.set_xlim(0, self.T)
        
        ax1.grid(True, axis='x', alpha=0.3)
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(self.times, self.lambda_values,
                            label=r"$\lambda^*(t)$", alpha=0.7, color='orange')
        ax2.set_ylabel(r"Intensité $\lambda^*(t)$")
        ax1.legend([line1, line2], ["N(t)", r"$\lambda^*(t)$"], loc="upper left")
        fig.tight_layout()
        plt.show()
    

class PoissonHomogeneous(PointProcess):
    def __init__(self, params):
        self.events = []
        self.T = params["T"]
        self.lambda_ = params["lambda"]
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate()

    def simulate(self):
        t = 0
        events = []
        while True:
            u = np.random.uniform(0, 1)
            w = -np.log(u) / self.lambda_
            t += w
            if t > self.T:
                break
            events.append(t)
        self.events = np.array(events)

class PoissonInhomogeneous(PointProcess):
    def __init__(self):
        self.events = []
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate()
          
    @staticmethod
    def lambda_t(t):
        return (np.sin(.2*t) + 1.5) ** 2
        
    def simulate(self):
        lambda_ = np.max(self.lambda_t(np.linspace(0, self.T, int(100*self.T))))
        poisson_h = PoissonHomogeneous({"T": self.T, "lambda": lambda_})
        events_h = poisson_h.events
        events = []
        for i in range(len(events_h)):
            u = np.random.uniform(0, 1)
            if u <= self.lambda_t(events_h[i]) / lambda_:
                events.append(events_h[i])
        self.events = np.array(events)
        
    
class HawkesExp(PointProcess):
    def __init__(self, params):
        print(self.__class__.__name__)
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
    def _intensity_on_grid(times, mu, alpha, beta, events):
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
    
class HawkesPL(PointProcess):
    def __init__(self, params):
        self.events = []
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
    def _intensity_on_grid(times, mu, alpha, beta, events):
        times = np.asarray(times, float)
        events = np.asarray(events, float)
        lam = np.full_like(times, mu, dtype=float)
        for k, t in enumerate(times):
            past = events[events < t]
            if past.size:
                lam[k] += alpha * np.sum((1.0 + (t - past))**(-(beta+1)))
        return lam
    

class HawkesMultiExp(PointProcess):
    def __init__(self, params):
        self.events = []
        self.T = params["T"]
        self.mu = params["mu"]
        self.alphas = np.array(params["alphas"])
        self.betas = np.array(params["betas"]) 
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.simulate_cluster() 

    def simulate_cluster(self):
        """
        Représentation en clusters : chaque événement génère des enfants
        selon un processus Poisson de paramètre alpha_j / beta_j pour chaque kernel j.
        """
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
    def _intensity_on_grid(times, mu, alphas, betas, events):
        times = np.asarray(times, float)
        events = np.asarray(events, float)
        lam = np.full_like(times, mu, dtype=float)

        for k, t in enumerate(times):
            past = events[events < t]
            if past.size:
                dt = t - past[:, None]     # shape (n_events, 1)
                kernels = alphas * np.exp(-betas * dt)
                lam[k] += kernels.sum()

        return lam

