import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

jr_eigen = np.load("AlAl_band_structure.npy")*27.2114
qe_eigen = np.load("qe.npy").T


# comparing jr and qe

plt.figure(figsize=(4, 3.6))
fermi = jr_eigen[18, 3]
plt.plot(jr_eigen - fermi, color="blue", alpha=0.3, lw=2)

# fermi = qe_eigen[18, 3]
fermi = 0
for i in range(qe_eigen.shape[1]):
  plt.scatter(
    range(60), qe_eigen[:, i]-fermi, color="brown", marker="v", alpha=1, s=3)

plt.vlines(18, 100, -100, linestyles="dotted", color="grey")
plt.vlines(40, 100, -100, linestyles="dotted", color="grey")
plt.hlines(0, 0, 60, linestyles="dotted", color="grey")
plt.xticks([0, 18, 40, 60], ["L", "$\Gamma$", "X", "L"])
plt.yticks([-20, -10, 0, 10, 20])
plt.ylabel("Eigenvalue (eV)")
plt.xlabel("K point")
plt.title("Al")

custom_handles = [
  Line2D([0], [0], color='blue', label="Our method"),
  Line2D(
    [0], [0], marker='v', color='w', markerfacecolor='brown', alpha=1,
    markersize=6, label="Quantum Espresso"
  )
]

# Custom legend
plt.legend(handles=custom_handles, loc='lower right')


plt.ylim(-30, 20)
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig("overlay.pdf")
print("overlay.pdf saved.")
plt.show()
