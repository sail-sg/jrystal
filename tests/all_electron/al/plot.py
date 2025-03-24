import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

jr_eigen = np.load("AlAlAlAl_band_structure.npy")*27.2114
qe_eigen = np.load("qe.npy").T


### comparing jr and qe

plt.figure(figsize=(4, 3.6))

fermi = jr_eigen[0, 25]
plt.plot(jr_eigen - fermi, color="blue", alpha = 0.3, lw=2)


fermi = 0
for i in range(qe_eigen.shape[1]):
    plt.scatter(range(60), qe_eigen[:, i]-fermi, color="brown", marker="v", alpha=1, s=3)

plt.xticks([0, 17, 34, 60], ["$\Gamma$", "X", "M", "$\Gamma$"])
plt.vlines(17, 100, -100, linestyles="dotted", color="grey")
plt.vlines(34, 100, -100, linestyles="dotted", color="grey")
plt.hlines(0, 0, 60, linestyles="dotted", color="grey")
plt.yticks([-10, -5, 0, 5, 10])
plt.ylabel("Eigenvalue (eV)")
plt.xlabel("K point")
plt.title("Al")

custom_handles = [
    Line2D([0], [0], color='blue', label="Our method"),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='brown', alpha=1, markersize=6, label="Quantum Espresso")
]

# Custom legend
plt.legend(handles=custom_handles, loc='lower right')


plt.ylim(-10, 10)
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig("overlay.pdf")
plt.show()

