from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float


def plot_band_structure(
  ks_eigs: Float[Array, "num_k num_bands"],
  figsize: Union[Tuple, List] = (5, 5),
  save_path: str = None,
  ylim: Tuple = None,
  title: str = "Band structure",
  **kwargs
):
  """Band Structure plot.

  TODO: Define a Bandpath object and support k-path label.

  Args:
      ks_eigs: Kohn-Sham eigenvalues in EV unit, already shifted s.t. fermi-level is zero.
      fermi_level: highest occupied energy level.
      figsize (Union[Tuple, List], optional): _description_. Defaults to (5, 5).
      save_path (str, optional): _description_. Defaults to None.
      ylim (Union[Tuple, List], optional): _description_. Defaults to None.
      title (str, optional): _description_. Defaults to "Band structure".

  Raises:
      ValueError: _description_

  Returns:
      _type_: _description_
  """
  valence_band = np.sum(np.max(ks_eigs, axis=0) <= 0)
  plt.rcParams["figure.figsize"] = figsize
  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111)
  ax.plot(ks_eigs[:, :valence_band], color="red", **kwargs)
  ax.plot(ks_eigs[:, valence_band:], color="blue", **kwargs)
  ax.axhline(0, 0, ks_eigs.shape[0], lw=0.2, color="black")
  ax.axhline(
    np.min(ks_eigs[:, valence_band + 1]),
    0,
    ks_eigs.shape[0],
    lw=0.2,
    color="black"
  )
  if ylim:
    ax.set_ylim(*ylim)
  ax.set_ylabel("E-E_fermi (eV)")
  ax.set_xlabel("k-path")
  ax.set_title(title)

  if save_path:
    fig.savefig(save_path)

  return fig.tight_layout()
