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

"""Implementation of the fixed point iteration method in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_structure

from jaxopt._src import base
from jaxopt._src.tree_util import tree_l2_norm, tree_sub


class FixedPointState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    value: pytree of current estimate of fixed point
    error: residuals of current estimate
    aux: auxiliary output of fixed_point_fun when has_aux=True
  """
  iter_num: int
  value: Any
  error: float
  aux: Any


@dataclass
class FixedPointIteration(base.IterativeSolver):
  """Fixed point iteration method.

  Attributes:
    fixed_point_fun: a function ``fixed_point_fun(x, *args, **kwargs)``
      returning a pytree with the same structure and type as x
      each leaf must be an array (not a scalar). The function
      should fulfill the Banach fixed-point theorem's assumptions.
      Otherwise convergence is not guaranteed.
    maxiter: maximum number of iterations.
    tol: tolerance (stopping criterion)
    has_aux: wether fixed_point_fun returns additional data. (default: False)
      if True, the fixed is computed only with respect to first element of the
      sequence returned. Other elements are carried during computation.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")

  References:
    https://en.wikipedia.org/wiki/Fixed-point_iteration
  """
  fixed_point_fun: Callable
  maxiter: int = 100
  tol: float = 1e-5
  has_aux: bool = False
  verbose: bool = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init(self,
           init_params,
           *args,
           **kwargs) -> base.OptStep:
    """Initialize the parameters and state.

    Args:
      init_params: initial guess of the fixed point, pytree
      *args: additional positional arguments to be passed to ``optimality_fun``.
      **kwargs: additional keyword arguments to be passed to ``optimality_fun``.
    Returns:
      (params, state)
    """
    state = FixedPointState(iter_num=0,
                            value=init_params,
                            error=jnp.inf,
                            aux=None)
    return base.OptStep(params=init_params, state=state)

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the fixed point iteration method.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to
        ``fixed_point_fun``.
      **kwargs: additional keyword arguments to be passed to
        ``fixed_point_fun``.
    Returns:
      (params, state)
    """
    next_params, aux = self._fun(params, *args, **kwargs)
    error = tree_l2_norm(tree_sub(next_params, params))
    next_state = FixedPointState(iter_num=state.iter_num + 1,
                                 value=next_params,
                                 error=error,
                                 aux=aux)
    return base.OptStep(params=next_params, state=next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    new_params, _ = self._fun(params, *args, **kwargs)
    return tree_sub(new_params, params)

  def __post_init__(self):
    if self.has_aux:
      self._fun = self.fixed_point_fun
    else:
      self._fun = lambda *a, **kw: (self.fixed_point_fun(*a, **kw), None)
