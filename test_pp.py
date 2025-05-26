"""
Test the pseudopotential file

projectors:
pp_dict['PP_NONLOCAL']['PP_BETA'][0]['values']
D_IJ:
pp_dict['PP_NONLOCAL']['PP_DIJ']: list
MULTIPOLES:
pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES']
Q_IJL:
pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ'][0]['values']

"""


import numpy as np
from matplotlib import pyplot as plt

from jrystal.pseudopotential.load import parse_upf


def main():
  # pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/normconserving/C.pz-vbc.UPF')
  # pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-rrkjus_psl.1.0.1.UPF")
  pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF")

  def int_over_grid(f):
    return np.sum(np.array(f) * np.array(pp_dict['PP_MESH']['PP_RAB']))

  _plot = False
  if _plot:
    """compare the AE_NLCC and NLCC"""
    index = 800
    s = 0
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][s:index],
      pp_dict['PP_NLCC'][s:index],
      label=r"$\widetilde{n}_c$"
    )
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][s:index],
      pp_dict['PP_PAW']['PP_AE_NLCC'][s:index],
      label=r"$n_c$"
    )
    plt.legend()
    plt.xlabel(r"$r$ (a.u.)")
    plt.ylabel(r"$\log n_c$ (a.u.)")
    plt.yscale("log")
    plt.savefig(f"fig/nlcc_cmp_{index}.png", dpi=300)
    plt.close()

    """compare the AE_WFC and PS_WFC"""
    index = 1000
    for i in range(4):
      plt.plot(
        pp_dict['PP_MESH']['PP_R'][:index],
        pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values'][:index],
        label=r"$\psi$"
      )
      plt.plot(
        pp_dict['PP_MESH']['PP_R'][:index],
        pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values'][:index],
        label=r"$\widetilde\psi$"
      )
      plt.legend()
      plt.xlabel(r"$r$ (a.u.)")
      plt.savefig(f"fig/wfc{i}_cmp.png", dpi=300)
      plt.close()

    """check the calculation of PP_QIJ
    TODO: it is weird that currectly Q_{ij}^L(r)/Q_{ij} are not the same for the
    same l and different i, j
    """
    index = 800
    i = 0
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][:index],
      np.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ'][i]['values'][:index])/
      pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES'][i],
      label=f"qij({i})"
    )
    i = 1
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][:index],
      np.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ'][i]['values'][:index])/
      pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES'][i],
      label=f"qij({i})"
    )
    i = 4
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][:index],
      np.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ'][i]['values'][:index])/
      pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES'][5],
      label=f"qij({i})"
    )
    plt.legend()
    plt.xlabel(r"$r$ (a.u.)")
    plt.savefig(f"fig/qij{i}_cmp.png", dpi=300)
    plt.close()

    """ compare the V_LOCAL and AE_V_LOCAL"""
    index = 800
    s = 0
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][s:index],
      -np.array(pp_dict['PP_LOCAL'][s:index]),
      label=r"$\widetilde{n}_c$"
    )
    plt.plot(
      pp_dict['PP_MESH']['PP_R'][s:index],
      -np.array(pp_dict['PP_PAW']['PP_AE_VLOC'][s:index]),
      label=r"$n_c$"
    )
    plt.legend()
    plt.xlabel(r"$r$ (a.u.)")
    plt.ylabel(r"$\log n_c$ (a.u.)")
    plt.yscale("log")
    plt.savefig(f"fig/vlocal_cmp_{index}.png", dpi=300)
    plt.close()

  """check the orthogonality of PS_WFC and BETA
  NOTE: the accuracy is relatively low comparing to other integrations
  """
  for i in range(2):
    for j in range(2):
      q_ij = int_over_grid(
        np.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *
        pp_dict['PP_NONLOCAL']['PP_BETA'][j]['values']
      )
      print(q_ij)
  for i in range(2, 4):
    for j in range(2, 4):
      q_ij = int_over_grid(
        np.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *
        pp_dict['PP_NONLOCAL']['PP_BETA'][j]['values']
      )
      print(q_ij)
      # assert (I[i, j] - q_ij) < 1e-8
  breakpoint()

  r"""check the calculation of D_{ij} factor, not work yet"""
  for i in range(2):
    for j in range(2):
      fr = np.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        pp_dict['PP_NONLOCAL']['PP_BETA'][j]['values']
      # the following line is incorrect according to the definition
      B_ij = np.sum(fr * np.array(pp_dict['PP_MESH']['PP_RAB']))
      q_ij = pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_Q'][i * 4 + j]
      D_ij = pp_dict['PP_NONLOCAL']['PP_DIJ'][i * 4 + j]
      qijr = np.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values']) *\
        np.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][j]['values']) -\
        np.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        np.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'])
      screen = int_over_grid(qijr * pp_dict['PP_PAW']['PP_AE_VLOC'])
      D_ij_ = D_ij + screen
      e_j = (D_ij_ - B_ij) / q_ij
      print(f"e{j}: {e_j:.4f}")
      # assert (I[i, j] - q_ij) < 1e-8


if __name__ == "__main__":
  main()
