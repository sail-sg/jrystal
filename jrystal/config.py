import ml_collections


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  config.crystal = 'diamond'
  config.ecut = 200
  config.grid_size = [20, 22, 24]

  config.k_grid_size = 1
  config.optimizer = 'sgd'
  config.optimizer_args = {'lr': 0.1, 'momentum': None}
  config.epoch = 5000
  config.convergence = 1e-5

  config.ewald_args = {'ew_eta': 0.08, 'ew_cut': 1e4, 'ewald_grid': None}

  return config
