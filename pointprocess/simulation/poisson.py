from .base import PointProcess
import numpy as np

class PoissonHomogeneous(PointProcess):
    def __init__(self, params):
        self.events = []
        self.params = params
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
    
    @staticmethod
    def _intensity_on_grid(times, params, events):
        return params["lambda"] * np.ones_like(times, dtype=float)
        
class PoissonInhomogeneous(PointProcess):
    def __init__(self, params):
        self.events = []
        self.params = params
        self.T = params["T"]
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

    def _intensity_on_grid(self, times, params, events):
        return self.lambda_t(times)