"""Module of visualization tools."""
import numpy as np
import matplotlib.pyplot as plt
from jrystal._src.occupation import get_fermi_level

from jaxtyping import Float, Array
from typing import Tuple, List, Union
from jrystal._src.const import HARTREE2EV


def plot_band_structure(
  eigenvalues: Float[Array, "num_k num_bands"],
  fermi_level: float = None,
  num_electrons: float = None,
  figsize: Union[Tuple, List] = (5, 5),
  save_path: str = None,
  ylim: Tuple = None,
  title: str = "Band structure",
  **kwargs
):
  """Band Structure plot.

  TODO: Define a Bandpath object and support k-path label.

  Args:
      eigenvalues (Float[Array, &quot;num_k num_bands&quot;]): _description_
      fermi_level (float, optional): _description_. Defaults to None.
      num_electrons (float, optional): _description_. Defaults to None.
      figsize (Union[Tuple, List], optional): _description_. Defaults to (5, 5).
      save_path (str, optional): _description_. Defaults to None.
      ylim (Union[Tuple, List], optional): _description_. Defaults to None.
      title (str, optional): _description_. Defaults to "Band structure".

  Raises:
      ValueError: _description_

  Returns:
      _type_: _description_
  """

  eigenvalues *= HARTREE2EV

  if fermi_level is None and num_electrons is None:
    raise ValueError(
      'Please indicate either the fermi level or number of', 'electrons'
    )
  elif fermi_level is None and num_electrons:
    fermi_level = get_fermi_level(eigenvalues, num_electrons // 2)

  eigenvalues -= fermi_level
  fermi_band = np.sum(np.max(eigenvalues, axis=0) <= 0)

  plt.rcParams["figure.figsize"] = figsize
  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111)
  ax.plot(eigenvalues[:, :fermi_band], color="red", **kwargs)
  ax.plot(eigenvalues[:, fermi_band:], color="blue", **kwargs)
  ax.axhline(0, 0, eigenvalues.shape[0], lw=0.2, color="black")
  ax.axhline(
    np.min(eigenvalues[:, fermi_band + 1]),
    0,
    eigenvalues.shape[0],
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


def plot_slice(
  data: np.ndarray, ncol: int = 10, cmap: str = 'coolwarm', same_scale=True
):
  """plot the slices of a 3d object

  Args:
    data (ndarray): the 3d array to plot
    ncol (int, optional): number of columns. Defaults to 10.
    cmap (str, optional): matplotlib colormaps.
  """

  nrow = int(np.ceil(data.shape[0] / ncol))
  cmap = 'coolwarm'

  plt.rcParams["figure.figsize"] = (ncol * 2, nrow * 2)
  fig, axs = plt.subplots(nrow, ncol)

  vmin = np.min(data) if same_scale else None
  vmax = np.max(data) if same_scale else None

  for idx, ax in enumerate(axs.flat[:data.shape[0]]):
    ax.imshow(data[idx].real, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()

  # Remove unused subplots
  for idx in range(data.shape[0], nrow * ncol):
    axs.flat[idx].set_visible(False)

  # fig.subplots_adjust(wspace=1, hspace=1)
  return fig.tight_layout()
