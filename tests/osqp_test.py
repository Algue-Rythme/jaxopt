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

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import test_util as jtu
import jax.numpy as jnp
from jax.test_util import check_grads
import numpy as onp

from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm

from jaxopt import projection
from jaxopt._src.base import KKTSolution
from jaxopt._src.osqp import OSQP
from jaxopt import QuadraticProgramming


#TODO(lbethune): utility for debuging, remove ?
def _print_jacob(solve_run_var, var, eps):
  zeros = jnp.zeros_like(var).ravel()
  for idx in range(var.size):
    dir = jnp.reshape(zeros.at[idx].set(1.), var.shape)
    delta = solve_run_var(var + eps*dir)-solve_run_P(var - eps*dir)
    deriv = delta / (2*eps)
    print(deriv)
  print(jax.jacrev(solve_run_var)(var))


def get_random_osqp_problem(problem_size, eq_constraints, ineq_constraints):
  assert problem_size >= eq_constraints  # very likely to be unfeasible
  P = onp.random.randn(problem_size, problem_size)
  P = P.T.dot(P)  # PSD matrix
  q = onp.random.randn(problem_size)
  A = onp.random.randn(ineq_constraints + eq_constraints, problem_size)
  l = onp.random.randn(ineq_constraints)
  u = l + jnp.abs(onp.random.randn(ineq_constraints))  # l < u
  b = onp.random.randn(eq_constraints)  # Ax = b
  l = jnp.concatenate([l, b])
  u = jnp.concatenate([u, b])
  return (P, q), A, (l, u)


def _from_osqp_form_to_default_form(P, q, A, l, u):
  is_eq = l == u
  is_ineq_l = jnp.logical_and(l != u, l != -jnp.inf)
  is_ineq_u = jnp.logical_and(l != u, u != jnp.inf)
  if jnp.any(is_eq):
    A_eq, b = A[is_eq,:], l[is_eq]
  else:
    A_eq, b = jnp.zeros((1,len(q))), jnp.zeros((1,))
  if jnp.any(is_ineq_l) and jnp.any(is_ineq_u):
    G = jnp.concatenate([-A[is_ineq_l,:],A[is_ineq_u,:]])
    h = jnp.concatenate([-l[is_ineq_l],u[is_ineq_u]])
  elif jnp.any(is_ineq_l):
    G, h = -A[is_ineq_l,:], -l[is_ineq_l]
  elif jnp.any(is_ineq_u):
    G, h = A[is_ineq_u,:], u[is_ineq_u]
  else:
    return (P, q), (A_eq, b), None
  return (P, q), (A_eq, b), (G, h)


class OSQPTest(jtu.JaxTestCase):

  @parameterized.product(eq_ineq=[(0, 4), (4, 0), (4, 4)])
  def test_small_qp(self, eq_ineq):
    # Setup a random QP min_x 0.5*x'*Q*x + q'*x s.t. Ax = z; l <= z <= u;
    eq_constraints, ineq_constraints = eq_ineq
    onp.random.seed(42)
    problem_size = 16
    params_obj, params_eq, params_ineq = get_random_osqp_problem(problem_size, eq_constraints, ineq_constraints)
    tol = 1e-5
    osqp = OSQP(tol=tol)
    init_params = osqp.init_params(params_obj, params_eq, params_ineq)
    params, state = osqp.run(init_params, params_obj, params_eq, params_ineq)
    self.assertLessEqual(state.error, tol)
    opt_error = osqp.l2_optimality_error(params, params_obj, params_eq, params_ineq)
    self.assertAllClose(opt_error, 0.0, atol=1e-3)

    def test_against_cvxpy(params_obj):
      (Q, c), Ab, Gh = _from_osqp_form_to_default_form(params_obj[0], params_obj[1],
                                                       params_eq, params_ineq[0], params_ineq[1])
      Q = 0.5 * (Q + Q.T)
      qp = QuadraticProgramming()
      hyperparams = dict(params_obj=(Q, c), params_eq=Ab, params_ineq=Gh)
      sol = qp.run(**hyperparams).params
      return sol.primal

    atol = 1e-4
    cvx_primal = test_against_cvxpy(params_obj)
    self.assertArraysAllClose(params.primal[0], cvx_primal, atol=atol)

    def osqp_run(params_obj):
      P, q = params_obj
      P = 0.5 * (P + P.T)
      sol = osqp.run(init_params, (P, q), params_eq, params_ineq).params
      return sol.primal[0]

    jacosqp = jax.jacrev(osqp_run)(params_obj)
    jaccvxpy = jax.jacrev(test_against_cvxpy)(params_obj)
    self.assertArraysAllClose(jacosqp[0], jaccvxpy[0], atol=5e-2)
    self.assertArraysAllClose(jacosqp[1], jaccvxpy[1], atol=5e-2)

  @parameterized.product(derivative_with_respect_to=[["none", "obj", "eq", "ineq"]])
  def test_qp_eq_and_ineq(self, derivative_with_respect_to):
    P = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    q = jnp.array([1.0, 1.0])
    A_eq = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])
    tol = 1e-5
    osqp = OSQP(tol=tol, verbose=0)

    @jax.jit
    def osqp_run(P, q, A_eq, G, b, h):
      l = jnp.concatenate([b, jnp.full(h.shape, -jnp.inf)])
      u = jnp.concatenate([b, jnp.full(h.shape, h)])
      P = 0.5 * (P + P.T) # we need to ensure that P is symmetric even after directional perturbations
      A = jnp.concatenate([A_eq, G], axis=0)
      init_params = osqp.init_params((P, q), A, (l, u))
      return osqp.run(init_params, (P, q), A, (l, u))

    atol = 1e-3
    eps = 1e-3

    if "none" in derivative_with_respect_to:
      params, state = osqp_run(P, q, A_eq, G, b, h)
      self.assertLessEqual(state.error, tol)
      assert state.status == OSQP.SOLVED

      qp = QuadraticProgramming()
      hyperparams_qp = dict(params_obj=(P, q), params_eq=(A_eq, b), params_ineq=(G, h))
      params_qp, state = qp.run(**hyperparams_qp)
      self.assertArraysAllClose(params.primal[0], params_qp.primal, atol=atol)

    def keep_ineq_only(params):
      b_idx = int(b.shape[0])
      mu, phi = params.dual_ineq
      return KKTSolution(params.primal, params.dual_eq, (mu[b_idx:], phi[b_idx:]))

    if "obj" in derivative_with_respect_to:
      solve_run_q = lambda q: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_q, args=(q,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_P = lambda P: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_P, args=(P,), order=1, modes=['rev'], eps=eps, atol=atol)

    if "eq" in derivative_with_respect_to:
      solve_run_A_eq = lambda A_eq: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_A_eq, args=(A_eq,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_b = lambda b: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_b, args=(b,), order=1, modes=['rev'], eps=eps, atol=atol)

    if "ineq" in derivative_with_respect_to:
      solve_run_G = lambda G: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_G, args=(G,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_h = lambda h: keep_ineq_only(osqp_run(P, q, A_eq, G, b, h).params)
      check_grads(solve_run_h, args=(h,), order=1, modes=['rev'], eps=eps, atol=atol)

  def test_projection_hyperplane(self):
    x = jnp.array([1.0, 2.0])
    a = jnp.array([-0.5, 1.5])
    b = 0.3
    q = -x
    # Find ||y-x||^2 such that jnp.dot(y, a) = b.

    matvec_P = lambda params_P,u: u
    matvec_A = lambda params_A,u: jnp.dot(a, u).reshape(1)

    tol = 1e-4
    osqp = OSQP(matvec_P=matvec_P, matvec_A=matvec_A, tol=tol, verbose=0)
    init_params = osqp.init_params((None, q), None, (b, b))
    sol, state = osqp.run(init_params, (None, q), None, (b, b))

    assert state.status == OSQP.SOLVED
    self.assertLessEqual(state.error, tol)
    atol = 1e-3
    opt_error = osqp.l2_optimality_error(sol, (None, q), None, (b, b))
    self.assertAllClose(opt_error, 0.0, atol=atol)
    expected = projection.projection_hyperplane(x, (a, b))
    self.assertArraysAllClose(sol.primal[0], expected, atol=atol)

  def test_projection_simplex(self):
    @jax.jit
    def _projection_simplex_qp(x, s=1.0):
      P = jnp.eye(len(x))
      A_eq = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      A = jnp.concatenate([A_eq, G])
      l = jnp.concatenate([b, jnp.full(h.shape, -jnp.inf)])
      u = jnp.concatenate([b, h])
      hyperparams = dict(params_obj=(P, -x), params_eq=A,
                         params_ineq=(l, u))

      osqp = OSQP(tol=1e-5)
      init_params = osqp.init_params(**hyperparams)
      return osqp.run(init_params, **hyperparams).params.primal[0]

    atol = 1e-3

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2, atol=atol)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=atol)

  def test_eq_constrained_qp_with_pytrees(self):
    rng = onp.random.RandomState(0)
    P = rng.randn(7, 7)
    P = onp.dot(P, P.T)
    A = rng.randn(4, 7)

    tmp = rng.randn(7)
    # Must have the same pytree structure as the output of matvec_P.
    q = {'foo':tmp[:3], 'bar':tmp[3:]}
    # Must have the same pytree structure as the output of matvec_A.
    b = [[rng.randn(1)] for _ in range(4)]

    def matvec_P(P, dic):
      x_ = jnp.concatenate([dic['foo'], dic['bar']])
      res = jnp.dot(P, x_)
      return {'foo':res[:3], 'bar':res[3:]}

    def matvec_A(A, dic):
      x_ = jnp.concatenate([dic['foo'], dic['bar']])
      z = jnp.dot(A, x_)
      ineqs = jnp.split(z,z.shape[0])
      return [[ineq] for ineq in ineqs]

    tol = 1e-5
    atol = 1e-4

    # With pytrees directly.
    hyperparams = dict(params_obj=(P, q), params_eq=A, params_ineq=(b, b))
    osqp = OSQP(matvec_P=matvec_P, matvec_A=matvec_A, tol=tol)
    # sol.primal has the same pytree structure as the output of matvec_P.
    # sol.dual_eq has the same pytree structure as the output of matvec_A.
    init_params = osqp.init_params(**hyperparams)
    sol_pytree, state = osqp.run(init_params, **hyperparams)
    assert state.status == OSQP.SOLVED
    self.assertAllClose(osqp.l2_optimality_error(sol_pytree, **hyperparams), 0.0, atol=atol)

    flat_x = lambda x: jnp.concatenate([x['foo'], x['bar']])
    flat_z = lambda z: jnp.concatenate([zi[0] for zi in z])

    # With flattened pytrees.
    q_flat = flat_x(q)
    b_flat = flat_z(b)
    hyperparams = dict(params_obj=(P, q_flat), params_eq=A, params_ineq=(b_flat, b_flat))
    osqp = OSQP(tol=tol)
    init_params = osqp.init_params(**hyperparams)
    sol = osqp.run(init_params, **hyperparams).params
    self.assertAllClose(osqp.l2_optimality_error(sol, **hyperparams), 0.0, atol=atol)

    # Check that the solutions match.
    self.assertArraysAllClose(flat_x(sol_pytree.primal[0]), sol.primal[0], atol=atol)
    self.assertArraysAllClose(flat_z(sol_pytree.primal[1]), sol.primal[1], atol=atol)
    self.assertArraysAllClose(flat_z(sol_pytree.dual_eq), sol.dual_eq, atol=atol)

  @parameterized.product(kernel=['linear','rbf'], size=[(50, 8), (400, 20)])
  def test_binary_kernel_svm(self, kernel, size):
    n_samples, n_features = size
    n_informative = (3*n_features) // 4
    # Prepare data.
    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                        n_informative=n_informative, n_classes=2,
                                        random_state=0)
    X = preprocessing.Normalizer().fit_transform(X)
    y = y * 2 - 1.  # Transform labels from {0, 1} to {-1, 1}.
    lam = 10.0
    C = 1./ lam

    if kernel == 'linear':
      K = jnp.dot(X, X.T)
    elif kernel == 'rbf':
      dists = jnp.expand_dims(X, 0) - jnp.expand_dims(X, 1)
      gamma = 1. / (X.shape[1] * jnp.var(X))  # like sklearn "scale" behavior
      K = jnp.exp(-gamma * jnp.sum(dists**2, axis=[-1]))

    # The dual objective is:
    # fun(beta) = 0.5 beta^T K beta - beta^T y
    # subject to
    # sum(beta) = 0
    # 0 <= beta_i <= C if y_i = 1
    # -C <= beta_i <= 0 if y_i = -1
    # where C = 1.0 / lam
    matvec_A = lambda _,b: (b, jnp.sum(b))
    l = -jax.nn.relu(-y * C), 0.
    u =  jax.nn.relu( y * C), 0.
    hyper_params = dict(params_obj=(K, -y), params_eq=None, params_ineq=(l, u))
    tol = 1e-2 if n_samples > 100 else 1e-5
    osqp = OSQP(matvec_A=matvec_A, tol=tol)

    sol = osqp.init_params(**hyper_params)
    sol, state = osqp.run(sol, **hyper_params)
    self.assertLessEqual(state.error, tol)
    atol = 0.5 if n_samples > 100 else 1e-2
    opt_error = osqp.l2_optimality_error(sol, **hyper_params)
    self.assertAllClose(opt_error, 0.0, atol=atol)

    def binary_kernel_svm_skl(K, y):
      svc = svm.SVC(kernel="precomputed", C=C, tol=tol).fit(K, y)
      dual_coef = onp.zeros(K.shape[0])
      dual_coef[svc.support_] = svc.dual_coef_[0]
      return dual_coef

    beta_fit_skl = binary_kernel_svm_skl(K, y)
    # we solve the dual problem with OSQP so the dual of svm.SVC
    # corresponds to the primal variables of OSQP solution.
    self.assertAllClose(sol.primal[0], beta_fit_skl, atol=5e-2)

  def test_infeasible_polyhedron(self):
    # argmin_p \|p - x\|_2 = argmin_p <p,Ip> - 2<x,p> = argmin_p 0.5pIp - <x,p>
    # under p1 + p2 =  1
    #            p2 =  1
    #      -p1 + p2 = -1
    #      -p1     <=  0
    #      -p2     <=  0
    # This QP is primal/dual infeasible.
    A = jnp.array([[1.0, 1.0],[0.0, 1.0],[-1.0, 1.0]])
    b = jnp.array([1.0, 1.0, -1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])

    l = b, jnp.full(h.shape, -jnp.inf)
    u = b, h

    def matvec_A(_, p):
      return jnp.dot(A, p), jnp.dot(G, p)

    x = jnp.zeros(2)
    I = jnp.eye(len(x))
    hyper_params = dict(params_obj=(I, -x), params_eq=None, params_ineq=(l, u))
    osqp = OSQP(matvec_A=matvec_A, check_primal_dual_infeasability=True, tol=1e-5)
    init_params = osqp.init_params(**hyper_params)
    sol, state = osqp.run(init_params, **hyper_params)
    assert state.status in [OSQP.PRIMAL_INFEASIBLE, OSQP.DUAL_INFEASIBLE]

  def test_infeasible_primal_only(self):
    # argmin   x1 + x2
    # under    x1 >= 6
    #          x2 >= 6
    #     x1 + x2 <= 11
    # This QP is primal infeasible.
    P = jnp.zeros((2,2))
    q = jnp.array([1.,1.])

    def matvec_A(_, x):
      return x[0], x[1], x[0] + x[1]
    l = 6., 6., -jnp.inf
    u = jnp.inf, jnp.inf, 11.
    
    hyper_params = dict(params_obj=(P, q), params_eq=None, params_ineq=(l, u))
    osqp = OSQP(matvec_A=matvec_A, check_primal_dual_infeasability=True)
    init_params = osqp.init_params(**hyper_params)
    sol, state = osqp.run(init_params, **hyper_params)
    assert state.status == OSQP.PRIMAL_INFEASIBLE

  def test_unbounded_primal(self):
    # argmin   x1 -2x2 + x3
    # under  x1 + x2 >= 0
    #              x3 = 1
    # This (degenerated) QP is dual infeasible (unbounded primal).
    def matvec_A(_, x):
      return x[0] + x[1], x[2]
    l = 0., 1.
    u = jnp.inf, 1.

    def matvec_P(_,x):
      return 0., 0., 0.

    hyper_params = dict(params_obj=(None, (-1., -2., 1.)), params_eq=None, params_ineq=(l, u))
    osqp = OSQP(matvec_P=matvec_P, matvec_A=matvec_A,
                check_primal_dual_infeasability=True,
                tol=1e-3,
                eq_qp_solve_tol=1e-8)  # seems necessary for dual infeasability detection
    init_params = osqp.init_params(x0=(5., -3., 0.02), **hyper_params)
    sol, state = osqp.run(init_params, **hyper_params)
    assert state.status == OSQP.DUAL_INFEASIBLE


if __name__ == '__main__':
  jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
