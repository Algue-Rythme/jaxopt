# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernels in matvec form."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp


class Kernel(ABC):
  """Base class for kernel."""

  def init_params(self, X, Y=None):
    """Return params from data matrices X and Y.

    Args:
      X: array of shape (num_samples_X, num_features).
      Y: (optional) array of shape (num_samples_Y, num_features).
         Default to X if not given.
    """
    if Y is None:
      return X, X
    return X, Y

  @abstractmethod
  def matvec(self, params, vec):
    """Dot product between vector and kernel."""
    pass

  def get_dense(self, params, vec):
    """Return dense matrix of Kernel from params and coefficients."""
    return jax.jacfwd(lambda x: self.matvec(params, x))(vec)


@dataclass(eq=False)
class LinearKernel(Kernel):
  """Linear kernel.

  .. math::
    k(x,y) = x^Ty

  num_samples: n
  num_features: k

  Single core complexity:
    Space: :math:`\mathbb{O}(nk)`.
    Time : :math:`\mathbb{O}(nk)`.

  Significant speedup can be observed on parallel architectures
  compared to this worst case complexity.
  """

  def matvec(self, params, vec):
    X, Y = params
    return jnp.dot(X, jnp.dot(Y.T, vec))

  def get_dense(self, params, vec):
    del vec
    X, Y = params
    return X @ Y.T


class ShiftInvariantKernel(Kernel):
  """Shift Invariant Kernel.

  .. math::
    k(x,y) = g(x-y)

  g must be symmetric: g(x-y)=g(y-x).

  num_samples: n
  num_features: k

  Single core complexity:
    Space: :math:`\mathbb{O}(nk)`.
    Time : :math:`\mathbb{O}(n^2k)`.

  Significant speedup can be observed on parallel architectures
  compared to this worst case complexity.

  Attributes:
    shift_invariant: the shift invariant function.
    memory_reduction: an int specifying by which factor to
      reduce memory consumption (default: 1).
      If kernel matrix has shape (n, m) it must be a divisor of n.
      Higher value may increase runtime but will run on device with less
      memory available.
  """
  def matvec(self, params, vec):

    X, Y = params

    def _row_vec(example, data, vec):
      example = jnp.expand_dims(example, axis=0)  # broadcasting
      # example.shape = (1, num_features)
      # data.shape = (num_samples, num_features)
      k_row = self.shift_invariant(example - data)
      # k_row.shape = (num_samples)
      return jnp.dot(k_row, vec)  # scalar

    _submat_vec = jax.vmap(_row_vec, in_axes=(0, None, None), out_axes=0)
    submat_vec = lambda smallX: _submat_vec(smallX, Y, vec)

    assert X.shape[0] % memory_reduction == 0
    inner = X.shape[0] // memory_reduction
    outer = memory_reduction

    X_reshaped = jnp.reshape(X, (outer, inner, X.shape[1]))
    K_reshaped =  jax.lax.map(submat_vec, X_reshaped) # map over vmap paradigm.
    K = jnp.reshape(K_reshaped, (inner * outer,) + K_reshaped.shape[2:])

    return K


@dataclass(eq=False)
class RBFKernel(ShiftInvariantKernel):
  """RBF kernel.

  .. math::
    k(x,y) = \exp{(-\gamma * \|x-y\|^2_2)}

  num_samples: n
  num_features: k

  Single core complexity:
    Space: :math:`\mathbb{O}(nk)`.
    Time : :math:`\mathbb{O}(n^2k)`.

  Significant speedup can be observed on parallel architectures
  compared to this worst case complexity.

  Attributes:
    gamma: float.
  """
  gamma: float
  memory_reduction: int = 1

  def __post_init__(self):
    def _rbf(delta):
      return jnp.exp(-self.gamma * jnp.sum(delta**2, axis=-1))
    self.shift_invariant = _rbf


@dataclass(eq=False)
class LaplacianKernel(ShiftInvariantKernel):
  """RBF kernel.

  .. math::
    k(x,y) = \exp{(-\gamma * \|x-y\|^2_2)}

  num_samples: n
  num_features: k

  Single core complexity:
    Space: :math:`\mathbb{O}(nk)`.
    Time : :math:`\mathbb{O}(n^2k)`.

  Significant speedup can be observed on parallel architectures
  compared to this worst case complexity.

  Attributes:
    gamma: float.
  """
  gamma: float
  memory_reduction: int = 1

  def __post_init__(self):
    def _laplacian(delta):
      return jnp.exp(-self.gamma * jnp.sum(jnp.abs(delta), axis=-1))
    self.shift_invariant = _rbf


@dataclass(eq=False)
class NystromKernel(Kernel):
  """Nystrom approximation.

  Approximate the true kernel matrix :math:`K` by
  a low rank matrix :math:`\bar{K}`.

  .math::

    \bar{K} = CW^{+}C^T

  where :math:`W^{+}` denotes the pseudo inverse of :math:`W=K_{:r,:r}`
  and :math:`C=K_{:n,:r}`.

  Note that :math:`\bar{K}` does not need to be materialized in memory to perform dot products.
  This is particularly useful on large datasets for which a matrix of size :math:`n^2` would not fit
  in memory.

  Warning: all rows of `K` are assumed to be sampled from the same distribution and in no particular order.
  If this assumption is not valid the algorithm will fail silently.
  Hence, we suggest to always shuffle data matrices before using NystromKernel.

  num_samples: n
  num_features: k
  rank: r < n (preferrably much lower).

  Single core complexity:
    Space: :math:`\mathbb{O}(nr+nk+r^2)`.
    Time (init_params): :math:`\mathbb{O}(r^2k+r^3+rnk)`.
    Time (matvec): :math:`\mathbb{O}(r^2+rn)`.

  Significant speedup can be observed on parallel architectures
  compared to this worst case complexity.

  Attributes:
    kernel: underlying kernel to approximate.
      It must exhibit ``matvec((X, Y), vec)`` and ``get_dense`` methods.
    rank: rank of the approximation.


  [1] http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2016/lecture9.pdf

  [2] Si, S., Hsieh, C.J. and Dhillon, I., 2016, June. Computationally efficient Nyström approximation using fast transforms.
  In International conference on machine learning (pp. 2655-2663). PMLR.

  [3] Hsieh, C.J., Si, S. and Dhillon, I.S., 2014. Fast prediction for large-scale kernel machines.
  In Advances in Neural Information Processing Systems (pp. 3689-3697).
  """
  kernel: Kernel
  rank: int

  def init_params(self, X, Y=None):
    """Nystrom parameters from data matrix X.

    No support for rectangular kernels: init_params(X,Y) is not supported.

    Args:
      X: data matrix of shape (num_samples, num_features).
    """
    assert Y is None
    
    Xr = X[:self.rank,:]  # shape (r, k)
    # Warning: truncation of r first rows of X is a sample from X
    # only if X's rows are iid. See docstring of the class.
    Krr_params = self.kernel.init_params(Xr)  # shape (r, k), (r, k)
    dummy_input = jnp.zeros(self.rank)  # shape r
    W = self.kernel.get_dense(Krr_params, dummy_input)  # shape (r, r)
    W_pinv = jnp.linalg.pinv(W)  # shape (r, r)
    C = self.kernel.get_dense((X, Xr), dummy_input)  # shape (n, r)
    return W_pinv, C, Xr

  def matvec(self, params, vec):
    W_pinv, C, _ = params
    # note that: C = K(X, Xr)
    y = jnp.dot(C, jnp.dot(W_pinv, jnp.dot(C.T, vec)))
    return y

  def get_predict_params(self, params, vec):
    W_pinv, C, Xr = params
    pred_coefs = jnp.dot(W_pinv, jnp.dot(C.T, vec))
    return pred_coefs, Xr

  def predict(self, predict_params, x_test):
    pred_coefs, Xr = predict_params
    return self.kernel.matvec((x_test, Xr), pred_coefs)
