from __future__ import annotations

import pytest

from jrystal.calc.backend import (
  AllElectronBackend,
  NormConservingBackend,
  get_backend,
)
from tests.smoke.helpers import make_config


def test_get_backend_returns_all_electron_by_default():
  config = make_config()
  config.method.use_pseudopotential = False

  backend = get_backend(config)

  assert isinstance(backend, AllElectronBackend)


def test_get_backend_returns_normconserving_when_enabled():
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "nc"

  backend = get_backend(config)

  assert isinstance(backend, NormConservingBackend)


def test_get_backend_rejects_unsupported_pseudopotential_type():
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "ultrasoft"

  with pytest.raises(NotImplementedError, match="Only norm-conserving"):
    get_backend(config)
