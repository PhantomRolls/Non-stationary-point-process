import numpy as np
import matplotlib.pyplot as plt

class PointProcess:
    def plot(self):
        self.lambda_values = self._intensity_on_grid(self.times, self.params, self.events)
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