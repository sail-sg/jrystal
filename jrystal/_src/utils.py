import jax
import jax.numpy as jnp
from typing import Callable, List, Dict
from jaxtyping import Complex, Float, Array
from absl import logging
import numpy as np


# TODO: the name doesn't reflect what the function does.
def vmapstack(times: int, args: List[Dict] = None) -> Callable:
  """mutiple times vmap over f, from the front to the end.

  Example:
    if f maps (3) -> (2)
    then vmapstack(f): (*batches, 3) -> (*batches, 2)

  Args:
      times (Int): number of vmap times. Must be the same as the dimension of
      the batches.
      args (List[Dict], optional): arguments for f. Defaults to None.

  Returns:
      Callable: a function that map from (*batches, _) to (*batches, _)
  """

  def decorator(f):
    if args:
      if len(args) != times:
        logging.error(
          f'the length of args ({len(args)}) is not the same '
          f'of times ({times}).'
        )

    for i in range(times):
      if args:
        f = jax.vmap(f, **args[i])
      else:
        f = jax.vmap(f)
    return f

  return decorator


def vmapstack_reverse(times: int) -> Callable:
  """Keep the vmap out axes at the end of the output
  arranged in 3d and the same order as the input.

  Example:
  if the original f maps `(3,) -> (2,)` and times = 3
  then output has `(n1, n2, n3, 3) -> (2, n1, n2, n3)`.

  Args:
      times (int): number of vmap times.

  Returns:
      Callable: _description_
  """

  args = [{'in_axes': i, 'out_axes': -i - 1} for i in range(times)]
  decorator = vmapstack(times, args)

  def _decorator(f):

    def wrapper(x: jax.Array):
      x = decorator(f)(x)
      assert x.ndim >= times, print(
        f"input array dim ({x.ndim}) cannot be"
        f"smaller than times ({times})"
      )

      axes = np.concatenate(
        (
          np.arange(x.ndim)[:(x.ndim - times)],
          np.arange(x.ndim)[-1:-1 - times:-1]
        )
      )
      return jnp.transpose(x, axes)

    return wrapper

  return _decorator


def complex_norm_square(x: Complex[Array, '...']) -> Float[Array, '...']:
  """Compute the Square of the norm of a complex number
  """
  return jnp.abs(jnp.conj(x) * x)


def quartile(n: int) -> List[int]:
  """Compute the quartiles of arange(n).

  Args:
      n (int): an integer

  Returns:
      List[int]: 3 quartiles.
  """
  quartiles = []
  for q in [0.25, 0.5, 0.75]:
    quartiles.append(
      int(np.quantile(np.arange(n) + 1, q, method="closest_observation"))
    )
  return quartiles


def get_fftw_factor(n: int):
  """ get fftw factor n = 2^a * 3^b * 5^c *7 ^d * 11^e * 13^f   and  e/f = 0/1
  prime_factor_list = [2, 3, 5, 7, 11, 13] smaller than 2049.
  """
  factor_list = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    27,
    28,
    30,
    32,
    33,
    35,
    36,
    39,
    40,
    42,
    44,
    45,
    48,
    49,
    50,
    52,
    54,
    55,
    56,
    60,
    63,
    64,
    65,
    66,
    70,
    72,
    75,
    77,
    78,
    80,
    81,
    84,
    88,
    90,
    91,
    96,
    98,
    99,
    100,
    104,
    105,
    108,
    110,
    112,
    117,
    120,
    125,
    126,
    128,
    130,
    132,
    135,
    140,
    143,
    144,
    147,
    150,
    154,
    156,
    160,
    162,
    165,
    168,
    175,
    176,
    180,
    182,
    189,
    192,
    195,
    196,
    198,
    200,
    208,
    210,
    216,
    220,
    224,
    225,
    231,
    234,
    240,
    243,
    245,
    250,
    252,
    256,
    260,
    264,
    270,
    273,
    275,
    280,
    286,
    288,
    294,
    297,
    300,
    308,
    312,
    315,
    320,
    324,
    325,
    330,
    336,
    343,
    350,
    351,
    352,
    360,
    364,
    375,
    378,
    384,
    385,
    390,
    392,
    396,
    400,
    405,
    416,
    420,
    429,
    432,
    440,
    441,
    448,
    450,
    455,
    462,
    468,
    480,
    486,
    490,
    495,
    500,
    504,
    512,
    520,
    525,
    528,
    539,
    540,
    546,
    550,
    560,
    567,
    572,
    576,
    585,
    588,
    594,
    600,
    616,
    624,
    625,
    630,
    637,
    640,
    648,
    650,
    660,
    672,
    675,
    686,
    693,
    700,
    702,
    704,
    715,
    720,
    728,
    729,
    735,
    750,
    756,
    768,
    770,
    780,
    784,
    792,
    800,
    810,
    819,
    825,
    832,
    840,
    858,
    864,
    875,
    880,
    882,
    891,
    896,
    900,
    910,
    924,
    936,
    945,
    960,
    972,
    975,
    980,
    990,
    1000,
    1001,
    1008,
    1024,
    1029,
    1040,
    1050,
    1053,
    1056,
    1078,
    1080,
    1092,
    1100,
    1120,
    1125,
    1134,
    1144,
    1152,
    1155,
    1170,
    1176,
    1188,
    1200,
    1215,
    1225,
    1232,
    1248,
    1250,
    1260,
    1274,
    1280,
    1287,
    1296,
    1300,
    1320,
    1323,
    1344,
    1350,
    1365,
    1372,
    1375,
    1386,
    1400,
    1404,
    1408,
    1430,
    1440,
    1456,
    1458,
    1470,
    1485,
    1500,
    1512,
    1536,
    1540,
    1560,
    1568,
    1575,
    1584,
    1600,
    1617,
    1620,
    1625,
    1638,
    1650,
    1664,
    1680,
    1701,
    1715,
    1716,
    1728,
    1750,
    1755,
    1760,
    1764,
    1782,
    1792,
    1800,
    1820,
    1848,
    1872,
    1875,
    1890,
    1911,
    1920,
    1925,
    1944,
    1950,
    1960,
    1980,
    2000,
    2002,
    2016,
    2025,
    2048
  ]

  factor_list = np.array(factor_list)
  if n > 2048:
    raise ValueError(
      f"The grid number {n} is too large, can't proceed calculation!"
    )
  delta_n = (factor_list - n) >= 0
  output = factor_list[delta_n][0]
  return output
