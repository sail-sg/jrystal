"""Error modules.

(follows Flax: https://github.com/google/flax/blob/main/flax/errors.py)

=== When to create an error class?

If an error message requires more explanation than a one-liner, it is useful to
add it as a separate error class. This may lead to some duplication with
existing documentation or docstrings, but it will provide users with more help
when they are debugging a problem. We can also point to existing documentation
from the error docstring directly.

=== How to name the error class?

* If the error occurs when doing something, name the error
  <Verb><Object><TypeOfError>Error

  For instance, if you want to raise an error when applying a module with an
  invalid method, the error can be: ApplyModuleInvalidMethodError.

 <TypeOfError> is optional, for instance if there is only one error when
  modifying a variable, the error can simply be: ModifyVariableError.

* If there is no concrete action involved the only a description of the error
 is sufficient. For instance: InvalidFilterError, NameInUseError, etc.


=== Copy/pastable template for new error messages:

class Template(JrystalError):
  "" "

  "" "
  def __init__(self):
    super().__init__(f'')
"""


class JrystalError(Exception):

  def __init__(self, message):
    super().__init__(message)


class InitiateQRDecompShapeError(JrystalError):
  """The QRdecomp module requires the input has a shape (..., M, K) where M>=K.
  This error is thrown when M < K.

  Example:

    >>> from jrystal import QRdecomp
    >>> shape = jnp.array([3, 4, 5])
    >>> qr = QRdecomp(shape)

    >>> jrystal.errors.InitiateQRdecompShapeError: The QRdecomp module requires
    the input has a shape(..., M, K) where M>=K. Got shape [3 4 5]

  """

  def __init__(self, shape):
    super().__init__(
      f"The QRdecomp module requires the input has a shape"
      f"(..., M, K) where M>=K. Got shape {shape}"
    )


class ApplyExpCoeffShapeError(JrystalError):
  """The ExpandCoeff module requires the self.mask module have the same number
  of masked element as the last dimension of coefficient array.

  Example:

    >>> cg = jnp.ones([2, 3, 5])
    >>> mask = jnp.ones([4, 5])
    >>> _coeff_expand(cg, mask)

    >>> jrystal.errors.ApplyExpCoeffShapeError: input coefficient has
    incompatible shapes ((2, 3, 5)) with mask (120 masked elements).

  """

  def __init__(self, cg_shape, mask_num):
    super().__init__(
      f'input coefficient has incompatible shapes ({cg_shape}) '
      f'with mask ({int(mask_num)} masked elements).'
    )


class ApplyFFTShapeError(JrystalError):
  """The input shape of fft module object should be incompatible with the
  dimension of the fft operation. This error is raised when the dim of input
  array is smaller than the fft dimension.

  Example:

  >>> from jrystal import BatchedFFT
  >>> fft = BatchedFFT(3)
  >>> x = jnp.ones([3, 5])
  >>> fft.apply(None, x)

  >>> jrystal.errors.ApplyFFTShapeError: Input array must have higher dimension
  than fft ndim 3. Got input shape: (3, 5).

  """

  def __init__(self, fft_dim, input_shape):
    super().__init__(
      f'Input array must have higher dimension than fft ndim '
      f'{fft_dim}. Got input shape: {input_shape}.'
    )


class WavevecOccupationMismatchError(JrystalError):
  """The wave function should return an jax.Array with shape: 
      wave_fun: r -> [nspin, ni, nk, ...] 
    The occupation mask array has a shape of [nspin, ni, nk].
    This error will return when mismatch

  Args:
      JrystalError (_type_): _description_
  """

  def __init__(self, wave_shape, occ_shape):
    super().__init__(
      f"Wave function shape ({wave_shape}) and occupation "
      f"shape ({occ_shape}) mismatch."
    )


if __name__ == "__main__":
  pass
