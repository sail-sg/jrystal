import jax
import jrystal
from jrystal import Crystal
from jrystal._src.modules import PlaneWaveDensity
from jrystal._src.pw import get_cg
from jrystal._src.argsdicts import PWDArgs

# Define by lattice vectors
a = 3.5667  # angstrom
lattice_vectors = [(0, a / 2, a / 2), (a / 2, 0, a / 2), (a / 2, a / 2, 0)]
symbols = 'C2'
positions = [(0, 0, 0), (a / 2, a / 2, a / 2)]
diamond = Crystal(
  symbols=symbols, positions=positions, lattice_vectors=lattice_vectors
)

# alternatively,
_pkg_path = jrystal.get_pkg_path()
diamond = Crystal(xyz_file=_pkg_path + '/geometries/diamond.xyz')

# Define the planewavefunction.
prm, grids = PWDArgs.get_PWD_args(diamond, 20, [20, 22, 24], 1)
r_grid, g_grid = grids

# define the model
pwd = PlaneWaveDensity(**prm)
key = jax.random.PRNGKey(123)
params = pwd.init(key, r_grid)
nr, ng = pwd.apply(params, r_grid)

# breakpoint()
print(nr.shape)

# Get cg
# w_re = params['params']['pw']['QRDecomp_0']['w_re']
# w_im = params['params']['pw']['QRDecomp_0']['w_im']

# w = w_re + 1.j * w_im
cg = pwd.get_cg(params)
print(cg.shape)

print(pwd.get_occ(params))
