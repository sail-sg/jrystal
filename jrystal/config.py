import ml_collections


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  config.seed = 123

  config.crystal = 'diamond'
  config.ecut = 100
  config.grid_size = 32
  config.k_grid_size = 1

  config.optimizer = 'yogi'
  config.optimizer_args = {'learning_rate': 1e-3}

  config.epoch = 10000
  config.convergence = 1e-6

  config.ewald_args = {'ew_eta': 0.1, 'ew_cut': 5e4, 'ewald_grid': None}

  return config
