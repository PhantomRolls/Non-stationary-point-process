import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

class PointProcess:
    def __init__(self, T):
        self.T = T
        self.events = np.array([])
        self.times = np.linspace(0, self.T, int(100*self.T))
        self.lambda_values = np.array([])

    def plot(self):
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
    def __init__(self, T, lambda_=1):
        super().__init__(T)
        self.lambda_ = lambda_
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
        self.lambda_values = np.full_like(self.times, self.lambda_)

class PoissonInhomogeneous(PointProcess):
    def __init__(self, T):
        super().__init__(T)
        self.simulate()
    
    @staticmethod
    def lambda_t(t):
        return np.sin(.2*t) + 1.5
        
    def simulate(self):
        lambda_ = np.max(self.lambda_t(np.linspace(0, self.T, int(100*self.T))))
        poisson_h = PoissonHomogeneous(self.T, lambda_)
        events_h = poisson_h.events
        events = []
        for i in range(len(events_h)):
            u = np.random.uniform(0, 1)
            if u <= self.lambda_t(events_h[i]) / lambda_:
                events.append(events_h[i])
        self.events = np.array(events)
        self.lambda_values = self.lambda_t(self.times)
                
class Hawkes(PointProcess):
    def __init__(self, T, params):
        super().__init__(T)
        self.mu = params["mu"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.simulate()
    
    def simulate(self):
        t = 0
        events = []
        lambda_t = self.mu
        M = self.mu
        self.lambda_values = np.array([lambda_t])
        G = 0
        while True:
            u = np.random.uniform(0, 1)
            w = -np.log(u) / M
            t += w
            if t > self.T:
                break
            D = np.random.uniform(0, 1)
            G_t = G * np.exp(-self.beta * w)
            lambda_t = self.mu + self.alpha * G_t
            if D <= lambda_t / M:
                events.append(t)
                self.lambda_values = np.append(self.lambda_values, lambda_t + self.alpha)
                M = lambda_t + self.alpha
                G = G_t + 1
            else:
                self.lambda_values = np.append(self.lambda_values, lambda_t)
                M = lambda_t
                G = G_t
        self.events = np.array(events)
        self.lambda_values = self._intensity_on_grid(self.times, self.mu, self.alpha, self.beta, self.events)
        
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


class SelfCorrecting(PointProcess):
    r"""Processus ponctuel auto-corrigeant Ogata (1988).

    L'intensité conditionnelle est donnée par :
    $$\lambda^*(t) = \exp(\mu + \beta t - \alpha N_t)$$
    où $N_t$ est le nombre d'événements strictement avant $t$.
    """

    _EPS = 1e-12

    def __init__(self, T, params):
        super().__init__(T)
        self.mu = params["mu"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        if self.beta <= 0:
            raise ValueError("Le paramètre 'beta' doit être strictement positif pour un processus auto-corrigeant.")
        self.simulate()

    def simulate(self):
        t = 0.0
        events = []
        while True:
            k = len(events)
            e = max(np.random.exponential(1.0), self._EPS)
            t_next = self._next_event_time(t, k, e)
            if t_next > self.T:
                break
            events.append(t_next)
            t = t_next
        self.events = np.array(events)
        self.lambda_values = self._intensity_on_grid(self.times, self.mu, self.alpha, self.beta, self.events)

    def _next_event_time(self, t, k, e):
        if self.beta < 1e-8:
            lam = np.exp(np.clip(self.mu - self.alpha * k, -700, 709))
            return t + e / lam
        b_term = np.log(self.beta) + np.log(e) - self.mu + self.alpha * k
        log_term = np.logaddexp(self.beta * t, b_term)
        return log_term / self.beta

    @staticmethod
    def _intensity_on_grid(times, mu, alpha, beta, events):
        lam = np.empty_like(times, dtype=float)
        k = 0
        n = len(events)
        for i, t in enumerate(times):
            while k < n and events[k] <= t:
                k += 1
            exponent = mu + beta * t - alpha * k
            lam[i] = np.exp(np.clip(exponent, -700, 709))
        return lam


class SelfCorrectingInhomogeneous(PointProcess):
    r"""Processus auto-corrigeant non stationnaire.

    L'intensité conditionnelle est :
    $$\lambda^*(t) = \exp\big(\mu + g(t) + \beta t - \alpha N_t\big)$$,
    où $g(t)$ est une fonction de dérive déterministe fournie par l'utilisateur.

    Paramètres attendus :

    - ``mu`` : composante constante du log-intensité.
    - ``alpha`` : impact correctif de chaque événement.
    - ``beta`` : composante linéaire (accélération) dans le temps.
    - ``baseline`` : fonction scalaire $g(t)$ (optionnelle, zéro par défaut).
    - ``initial_window`` : fenêtre de recherche initiale pour l'événement suivant.
    - ``max_extension`` : extension maximale lors de la recherche de racine.
    """

    _EPS = 1e-12

    def __init__(self, T, params):
        super().__init__(T)
        self.mu = params["mu"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        if self.beta <= 0:
            raise ValueError("Le paramètre 'beta' doit être strictement positif pour un processus auto-corrigeant.")
        baseline = params.get("baseline")
        if baseline is None:
            self._baseline = lambda t: 0.0
        elif callable(baseline):
            self._baseline = baseline
        else:
            raise TypeError("Le paramètre 'baseline' doit être une fonction scalaire de t.")
        self.initial_window = max(params.get("initial_window", 1.0), 1e-6)
        self.max_extension = max(params.get("max_extension", max(5.0, 0.25 * self.T)), 1.0)
        self.simulate()

    def simulate(self):
        t = 0.0
        events = []
        while True:
            k = len(events)
            e = max(np.random.exponential(1.0), self._EPS)
            t_next = self._next_event_time(t, k, e)
            if not np.isfinite(t_next) or t_next > self.T:
                break
            events.append(t_next)
            t = t_next
        self.events = np.array(events)
        self.lambda_values = self._intensity_on_grid(
            self.times, self.mu, self.alpha, self.beta, self._baseline, self.events
        )

    def _next_event_time(self, t, k, target):
        # Recherche du prochain événement via inversion numérique de la fonction intensité cumulée.
        upper = t + self.initial_window
        integral = self._integrated_intensity(t, upper, k)
        # Étend la fenêtre jusqu'à dépasser la quantité cible.
        while integral < target:
            if upper - t > self.max_extension or upper >= self.T + self.max_extension:
                return np.inf
            step = max(self.initial_window, 0.5 * (upper - t))
            upper += step
            integral = self._integrated_intensity(t, upper, k)
        if integral <= 0:
            return np.inf

        def root_func(x):
            return self._integrated_intensity(t, x, k) - target

        lower = max(t + self._EPS, t)
        try:
            return brentq(root_func, lower, upper, xtol=1e-10, rtol=1e-10, maxiter=100)
        except ValueError:
            return np.inf

    def _integrated_intensity(self, start, end, k):
        if end <= start:
            return 0.0

        alpha_term = -self.alpha * k

        def integrand(s):
            exponent = self.mu + self.beta * s + self._baseline_value(s) + alpha_term
            exponent = np.clip(exponent, -700, 709)
            return np.exp(exponent)

        val, _ = quad(integrand, start, end, epsabs=1e-9, epsrel=1e-7, limit=100)
        return max(val, 0.0)

    def _baseline_value(self, t):
        return float(self._baseline(t))

    @staticmethod
    def _intensity_on_grid(times, mu, alpha, beta, baseline_func, events):
        lam = np.empty_like(times, dtype=float)
        k = 0
        n = len(events)
        for i, t in enumerate(times):
            while k < n and events[k] <= t:
                k += 1
            exponent = mu + beta * t + float(baseline_func(t)) - alpha * k
            exponent = np.clip(exponent, -700, 709)
            lam[i] = np.exp(exponent)
        return lam

