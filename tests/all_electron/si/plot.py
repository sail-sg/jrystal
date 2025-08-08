import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

jr_eigen = np.load("SiSi_band_structure.npy")*27.2114
jr_eigen = jr_eigen[:, 0] # remove spin channel

qe_eigen = np.load("qe.npy").T



### comparing jr and qe

plt.figure(figsize=(4, 3.6))

fermi = np.max(jr_eigen, axis=0)[13]
plt.plot(jr_eigen - fermi, color="blue", alpha = 0.3, lw=2)

fermi = np.max(qe_eigen, axis=0)[13]
for i in range(qe_eigen.shape[1]):
    plt.scatter(range(60), qe_eigen[:, i]-fermi, color="brown", marker="v", alpha=1, s=3)

plt.vlines(18, 100, -100, linestyles="dotted", color="grey")
plt.vlines(40, 100, -100, linestyles="dotted", color="grey")
plt.hlines(0, 0, 60, linestyles="dotted", color="grey")
plt.xticks([0, 18, 40, 60], ["L", "$\Gamma$", "X", "L"])
plt.yticks([-20, -10, 0, 10, 20])
plt.ylabel("Eigenvalue (eV)")
plt.xlabel("K point")
plt.title("Si")

custom_handles = [
    Line2D([0], [0], color='blue', label="Our method"),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='brown', alpha=1, markersize=6, label="Quantum espresso")
]

# Custom legend
plt.legend(handles=custom_handles, loc='lower right')

plt.ylim(-10, 10)
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig("overlay.pdf")
