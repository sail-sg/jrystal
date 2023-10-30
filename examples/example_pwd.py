import jax
from jrystal import Crystal
from jrystal._src.module import PlaneWaveDensity
from jrystal._src.pw import get_plane_wave_params


# Define by lattice vectors
a = 3.5667   # angstrom
lattice_vectors = [(0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]
symbols = 'C2'
positions = [(0, 0, 0), (a/2, a/2, a/2)] 
diamond = Crystal(symbols=symbols, positions=positions,
                  lattice_vectors=lattice_vectors)

# Define the planewavefunction.
prm, vec = get_plane_wave_params(diamond, 20, [20, 22, 24], 1)
r_vec, g_vec = vec
pwd = PlaneWaveDensity(*prm)
key = jax.random.PRNGKey(123)
params = pwd.init(key, r_vec)
nr, ng = pwd.apply(params, r_vec)

print(nr.shape)
