import ml_collections


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  config.seed = 123

  config.crystal = 'diamond'
  config.ecut = 400
  config.grid_size = 96
  config.k_grid_size = 1

  config.optimizer = 'yogi'
  config.optimizer_args = {'learning_rate': 1e-5}

  config.epoch = 5000
  config.convergence = 1e-6

  config.ewald_args = {'ew_eta': 0.08, 'ew_cut': 1e4, 'ewald_grid': None}

  return config
