import numpy as np
from ase.build import bulk

from gpaw import GPAW, PW
from matplotlib import pyplot as plt

name = 'C-diamond'
a = 3.5668  # diamond lattice parameter in Angstrom
method = 'direct' # 'scf' or 'direct'

bulk = bulk('C', 'diamond', a=a)

k = 1

cut_off_list = [400, 800, 1600, 3200, 6400]
timers = {k: [] for k in ['XC 3D grid', 'XC Correction', 'Potential matrix',
                           'Atomic hamiltonian', 'Mix', 'Total']}
for cut_off in cut_off_list:
  print(f"\nRunning calculation with cutoff energy: {cut_off} eV")
  if method == 'scf':
    calc = GPAW(
      mode=PW(cut_off, force_complex_dtype=True),
      setups='paw',
      kpts=(k, k, k),
      xc='PBE',
      symmetry='off',
      convergence={'energy': 1e-6},
      txt=name + '.txt'
    )
  elif method == 'direct':
    calc = GPAW(
      mode=PW(cut_off, force_complex_dtype=True),
      setups='paw',
      kpts=(k, k, k),
      xc='PBE',

      # Direct optimization settings
      eigensolver={'name': 'etdm-fdpw'},
      mixer={'backend': 'no-mixing'},
      occupations={'name': 'fixed-uniform'},
      symmetry='off',

      convergence={'eigenstates': 1e-6},
      txt=name + '.txt'
    )

  bulk.calc = calc
  bulk.get_potential_energy()

  t = calc.timer.timers
  if method == 'scf':
    timers['XC 3D grid'].append(t[('SCF-cycle', 'Hamiltonian', 'XC 3D grid')])
    timers['XC Correction'].append(t[('SCF-cycle', 'Hamiltonian', 'Atomic', 'XC Correction')])
    timers['Potential matrix'].append(t[('LCAO initialization', 'LCAO eigensolver', 'Potential matrix')])
    timers['Atomic hamiltonian'].append(t[('SCF-cycle', 'Hamiltonian', 'Calculate atomic Hamiltonians')])
    timers['Mix'].append(t.get(('SCF-cycle', 'Density', 'Mix'), (0,)))
    timers['Total'].append(calc.timer.get_time('SCF-cycle'))
  elif method == 'direct':
    timers['XC 3D grid'].append(t[('SCF-cycle', 'Direct Minimisation step', 'Hamiltonian', 'XC 3D grid')])
    timers['XC Correction'].append(t[('SCF-cycle', 'Direct Minimisation step', 'Hamiltonian', 'Atomic', 'XC Correction')])
    timers['Potential matrix'].append(t[('LCAO initialization', 'LCAO eigensolver', 'Potential matrix')])
    timers['Atomic hamiltonian'].append(t[('SCF-cycle', 'Direct Minimisation step', 'Hamiltonian', 'Calculate atomic Hamiltonians')])
    timers['Mix'].append(t.get(('SCF-cycle', 'Direct Minimisation step', 'Density', 'Mix'), (0,)))
    timers['Total'].append(calc.timer.get_time('SCF-cycle'))

for name, times in timers.items():
  plt.plot(cut_off_list, times, 'o-', label=name)
plt.plot(np.array(cut_off_list), np.array(cut_off_list) * np.log(np.array(cut_off_list)), 'k--', label=r'$O(N\log N)$')
plt.xlabel('Cutoff energy (eV)')
plt.ylabel('Time (s)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("profile_cutoffs.png")
