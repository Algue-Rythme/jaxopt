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

"""
Implicit differentiation of linear optimal control with CVXPY.
==============================================================

The objective function is a linear optimal control problem with a quadratic cost.
The problem is solved by solving a convex optimization problem with CVXPY, following their
original's implementation:
    https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/intro/control.ipynb

We use implicit differentiation to comptue the derivative of the control with
respect do dynamic's parameters A and B.

We have a system with a state $x_t\in {\bf R}^n$ that varies over the time steps $t=0,\ldots,T$,
and inputs or actions $u_t\in {\bf R}^m$ we can use at each time step to affect the state.
For example, $x_t$ might be the position and velocity of a rocket and $u_t$ the output of the rocket's thrusters.
We model the evolution of the state as a linear dynamical system, i.e.,

$$ x_{t+1} = Ax_t + Bu_t $$

where $A \in {\bf R}^{n\times n}$ and $B \in {\bf R}^{n\times m}$ are known matrices.

Our goal is to find the optimal actions $u_0,\ldots,u_{T-1}$ by solving the optimization problems

\begin{array}{ll} \mbox{minimize} & \sum_{t=0}^{T-1} \ell (x_t,u_t) + \ell_T(x_T)\\
\mbox{subject to} & x_{t+1} = Ax_t + Bu_t\\%, \quad t=0, \ldots, T-1\\
& (x_t,u_t) \in \mathcal C, \quad x_T\in \mathcal C_T,
%, \quad \quad t=0, \ldots, T
\end{array}

where $\ell: {\bf R}^n \times {\bf R}^m\to {\bf R}$ is the stage cost, $\ell_T$ is the terminal cost,
$\mathcal C$ is the state/action constraints, and $\mathcal C_T$ is the terminal constraint.
The optimization problem is convex if the costs and constraints are convex.

In the following code we solve a control problem with $n=8$ states, $m=2$ inputs, and horizon $T=50$.
The matrices $A$ and $B$ and the initial state $x_0$ are randomly chosen (with $A\approx I$).
We use the (traditional) stage cost $\ell(x,u) = \|x\|_2^2 + \|u\|_2^2$, 
the input constraint $\|u_t\|_\infty \leq 1$, and the terminal constraint $x_{T}=0$.
"""

from absl import app
import jax
import jax.numpy as jnp
import numpy as onp
from jaxopt import implicit_diff
from jaxopt.base import KKTSolution
import cvxpy as cp


def generate_problem_data(n, m, T, alpha, beta):
  A = onp.eye(n) + alpha*onp.random.randn(n,n)
  B = onp.random.randn(n,m)
  x_0 = beta*onp.random.randn(n)
  return x_0, A, B


def solve_optimal_control_problem(x_0, A, B, T):

  def obj_fun(primal, obj_params):
    del obj_params  # Unused here;
    x, u = primal
    return jnp.sum(x[:,1:]**2) + jnp.sum(u**2)

  def eq_fun(primal, eq_params):
    x, u = primal
    A, B = eq_params
    res  = x[:,1:] - (A @ x[:,:-1] + B @ u)
    return res

  def ineq_fun(primal, ineq_params):
    del ineq_params  # Unused here;
    _, u = primal
    res  = jnp.maximum(jnp.abs(u), axis=0) - 1
    return res

  optimality_fun = implicit_diff.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)

  # The body of this function is non differentiable, so we need to use a custom root.
  # solve_with_cvxpy must have the same input signature as optimality_fun
  @implicit_diff.custom_root(optimality_fun)  
  def solve_with_cvxpy(init_params, obj_params, eq_params, ineq_params):
    del init_params  #unused but present to respect signature

    # Define the problem
    A, B, T = eq_params
    n, m = A.shape[0], B.shape[1]
    x = cp.Variable((n, T+1))
    u = cp.Variable((m, T))
    # the pair (x,u) is the state and action at each time step,
    # and also primal variables.
    
    cost = 0
    eq_constr = [x[:,0] == x_0]
    ineq_constr = []
    for t in range(T):
        cost += cp.sum_squares(x[:,t+1]) + cp.sum_squares(u[:,t])
        eq_constr.append(x[:,t+1] == A@x[:,t] + B@u[:,t])
        ineq_constr.append(cp.norm(u[:,t], 'inf') <= 1)
    # Sums problem objectives and concatenates constraints.
    eq_constr += [x[:,T] == 0]
    problem = cp.Problem(cp.Minimize(cost), eq_constr+ineq_constr)
    problem.solve(solver=cp.ECOS)

    # Retrieve primal/dual variables
    primal         = (x.value, u.value)
    # We transform the list of dual variables into a single matrix. 
    duals_var_eq   = onp.stack([c.dual_value() for c in eq_constr], axis=-1)
    duals_var_ineq = onp.stack([c.dual_value() for c in ineq_constr], axis=-1)
    return KKTSolution(primal, duals_var_eq, duals_var_ineq)

  kkt_sol = solve_with_cvxpy(None, None, (A, B, T), None)
  return kkt_sol.primal


onp.random.seed(1)
n = 8
m = 2
alpha = 0.2
beta = 5
x_0, A, B = generate_problem_data(n, m, alpha, beta)
T = 50
x, u = solve_optimal_control_problem(x_0, A, B, T)