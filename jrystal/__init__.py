# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.0.1"

from pathlib import Path

from . import (
  _src,
  calc,
  config,
  crystal,
  energy,
  entropy,
  ewald,
  grid,
  hamiltonian,
  occupation,
  potential,
  pseudopotential,
  pw,
  sbt,
  utils
)
from .crystal import Crystal


def get_pkg_path():
  return str(Path(__file__).parent.parent)


__all__ = [
  "_src",
  "crystal",
  "Crystal",
  "calc",
  "pseudopotential",
  "utils",
  "sbt",
  "pw",
  "occupation",
  "grid",
  "hamiltonian",
  "energy",
  "potential",
  "entropy",
  "get_pkg_path",
  "config",
  "ewald",
]
