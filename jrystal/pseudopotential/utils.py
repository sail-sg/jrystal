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

from functools import wraps
from typing import List, Callable, Any
import jax
import numpy as np
import jax.numpy as jnp


def map_over_atoms(fun: Callable[..., Any]) -> Callable[..., List]:
  """A wrapper function that extends a single-atom operation to apply to all
    atoms in a collection.

    If f is a function that operates on an array x representing a single atom,
    then map_over_atoms(f) transforms f to operate on a list of arrays x_list,
    applying the function f to each array in the list, representing a series of
    atoms, and return a list of the outputs for all the atoms.

  Args:
      fun (Callable[..., Any]): _description_

  Returns:
      Callable: _description_
  """

  @wraps(fun)
  def map_fun(*arrays, **kwargs):

    num_atoms = len(arrays[0])
    output = [
      fun(*(arr[i] for arr in arrays), **kwargs) for i in range(num_atoms)
    ]
    return output

  return map_fun


def stack_with_padding(array_list: List[List]):
  # Input validation
  if not array_list:
    raise ValueError("Input list is empty!")

  # Verify all arrays have same number of dimensions
  ndim = array_list[0].ndim
  for arr in array_list:
    if arr.ndim != ndim:
      raise ValueError("All arrays must have the same number of dimensions!")

  # Compute maximum length for each dimension
  max_shape = np.max([arr.shape for arr in array_list], axis=0)

  # Pad each array only where dimensions don't match
  padded_arrays = []
  for arr in array_list:
    # Determine padding for each axis:
    # - (0, pad_size) for mismatched axes
    # - (0, 0) for already matching axes
    pad_width = []
    for axis in range(ndim):
      if arr.shape[axis] != max_shape[axis]:
        pad_width.append((0, max_shape[axis] - arr.shape[axis]))
      else:
        pad_width.append((0, 0))
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    padded_arrays.append(padded_arr)

  # Stack all arrays
  return np.stack(padded_arrays)
