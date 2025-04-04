import matplotlib.pyplot as plt
import LeMoC as lc
from simulation_params import SimulationParams
import numpy as np

if __name__ == "__main__":
    simpam = SimulationParams()
    so = lc.run(simpam)
    y = np.array(so.Spec_temp_tot)
    x = np.array(so.nu_tot)

    fig,ax = plt.subplots()

    iteration = 1
    colors = plt.cm.viridis(np.linspace(0, 1, y.shape[0]))

    for i in range(0, y.shape[0], iteration):
        color = colors[i]
        ax.plot(x, y[i, :], color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([1e10, 1e27])
    ax.set_ylim([1e33, 1e40])
    ax.legend()

    plt.show()