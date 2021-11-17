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
"""Interface for linear operators."""

import functools
import jax
import jax.numpy as jnp
import numpy as onp
from jax.experimental import sparse

from jaxopt.tree_util import tree_map, tree_sum, tree_mul


class DenseLinearOperator:

  def __init__(self, weights):
    self.weights = weights

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    return tree_map(jnp.dot, self.weights, x)

  def rmatvec(self, _, y):
    return tree_map(jnp.dot, self.weights.T, y)

  def matvec_and_rmatvec(self, x, y):
    return self.matvec(x), self.rmatvec(x, y)

  def normal_matvec(self, x):
    """Computes A^T A x from matvec(x) = A x, where A is square."""
    return self.rmatvec(x, self.matvec(x))

  def diag(self):
    diags_only = tree_map(jnp.diag, self.weights)
    return diags_only

  def columns_l2_norms(self, squared=False):
    def col_norm(w):
      col_norms = jnp.sum(jnp.square(w), axis=0)
      if not squared:
        col_norms = jnp.sqrt(col_norms)
      return col_norms
    return tree_map(col_norm, self.weights)


class SparseLinearOperator:

  def __init__(self, weights):
    self.weights = weights

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    def is_leaf(leaf):
      return isinstance(leaf, sparse.BCOO)
    return tree_map(sparse.sparsify(jnp.dot), self.weights, x, is_leaf=is_leaf)

  def rmatvec(self, _, y):
    def is_leaf(leaf):
      return isinstance(leaf, sparse.BCOO)
    return tree_map(lambda w,yi: sparse.sparsify(jnp.dot)(w.T,yi), self.weights, x, is_leaf=is_leaf)

  def matvec_and_rmatvec(self, x, y):
    return self.matvec(x), self.rmatvec(x, y)

  def normal_matvec(self, x):
    """Computes A^T A x from matvec(x) = A x, where A is square."""
    return self.rmatvec(x, self.matvec(x))

  def diag(self):
    diags_only = tree_map(jnp.diag, self.weights)
    return diags_only

  def columns_l2_norms(self, squared=False):
    def col_norm(w):
      col_norms = jnp.sum(jnp.square(w), axis=0)
      if not squared:
        col_norms = jnp.sqrt(col_norms)
      return col_norms
    return tree_map(col_norm, self.weights)


class FunctionalLinearOperator:

  def __init__(self, fun, params):
    self.fun = functools.partial(fun, params)

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    return self.fun(x)

  def rmatvec(self, x, y):
    return self.matvec_and_rmatvec(x, y)[1]

  def matvec_and_rmatvec(self, x, y):
    matvec_x, vjp = jax.vjp(self.matvec, x)
    rmatvec_y, = vjp(y)
    return matvec_x, rmatvec_y

  def normal_matvec(self, x):
    """Computes A^T A x from matvec(x) = A x, where A is square."""
    matvec_x, vjp = jax.vjp(self.matvec, x)
    return vjp(matvec_x)[0]
