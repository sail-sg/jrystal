import jrystal
from jrystal.crystal import Crystal

_pkg_path = jrystal.get_pkg_path()

# Define a diamond structure

# Define by lattice vectors
a = 3.5667  # angstrom
cell_vectors = [(0, a / 2, a / 2), (a / 2, 0, a / 2), (a / 2, a / 2, 0)]
symbols = 'C2'
positions = [(0, 0, 0), (a / 2, a / 2, a / 2)]
diamond = Crystal(
  symbols=symbols, positions=positions, cell_vectors=cell_vectors
)

print("symbols: ", diamond.symbols)
print("cell vectors: ", diamond.A)
print("reciprocal vectors: ", diamond.B)
print("unit cell volume: ", diamond.vol)

# Define by xyz file
diamond = Crystal(xyz_file=_pkg_path + '/geometries/diamond.xyz')

print("symbols: ", diamond.symbols)
print("cell vectors: ", diamond.A)
print("reciprocal vectors: ", diamond.B)
print("unit cell volume: ", diamond.vol)
