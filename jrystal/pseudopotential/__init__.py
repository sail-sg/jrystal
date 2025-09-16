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
"""The pseudopotential module.

.. warning::
The pseudopotential module is currently under development and may
undergo changes in future versions. At this time, we only support the UPF
format. Additionally, our implementation is limited to
norm-conserving pseudopotentials. Please note that many functions in this module
are not yet fully differentiable.

"""
from . import beta, dataclass, load, local, nloc, normcons, spherical, utils
from .dataclass import NormConservingPseudopotential, Pseudopotential

__all__ = [
  "dataclass",
  "Pseudopotential",
  "NormConservingPseudopotential",
  'local',
  'load',
  'spherical',
  'beta',
  'utils',
  'normcons',
  'nloc',
]
