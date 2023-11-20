import jax
import jrystal
from jrystal.crystal import Crystal
from jrystal.wave import PlaneWaveDensity
from jrystal._src.paramdict import PWDArgs

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
