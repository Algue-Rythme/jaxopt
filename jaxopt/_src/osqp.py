from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from dataclasses import dataclass
from functools import partial

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.tree_util import tree_reduce

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.tree_util import tree_add, tree_sub, tree_mul, tree_scalar_mul
from jaxopt._src.tree_util import tree_neg, tree_add_scalar_mul
from jaxopt._src.tree_util import tree_map, tree_vdot, tree_l2_norm, tree_norm_inf
from jaxopt._src.tree_util import tree_ones_like, tree_zeros_like, tree_where
from jaxopt._src.linear_solve import solve_normal_cg
from jaxopt._src.quadratic_prog import _matvec_and_rmatvec
from jaxopt.projection import projection_box


def _make_osqp_optimality_fun(matvec_P, matvec_A):
  """Makes the optimality function for OSQP.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, ineq_dual_var)
      params_obj = (P, q)
      params_eq = A
      params_ineq = (l, u)
  """
  def obj_fun(primal_var, params_obj):
    x, _ = primal_var
    params_P, q = params_obj
    P = matvec_P(params_P)
    # minimize 0.5xPx + qx
    qp_obj = tree_add_scalar_mul(tree_vdot(q, x), 0.5, tree_vdot(x, P(x)))
    return qp_obj

  def eq_fun(primal_var, params_eq):
    # constraint Ax=z associated to y^T(Ax-z)=0
    x, z = primal_var
    A = matvec_A(params_eq)
    z_bar = A(x)
    return tree_sub(z_bar, z)

  def ineq_fun(primal_var, params_ineq):
    # constraints l<=z<=u associated to mu^T(z-u) + phi^T(l-z)
    _, z = primal_var
    l, u = params_ineq
    # if l=-inf (resp. u=+inf) then phi (resp. mu)
    # will be zero (never active at infinity)
    # so we may set the residual l-z (resp. z-u) to zero too.
    # since 0 * inf = 0 here.
    # but not in IEEE 754 standard where 0 * inf = nan.
    # Note: the derivative in +inf or -inf does not make sense anyway,
    # but those terms need to be computed since they are part of Lagrangian.
    u_inf = tree_map(lambda ui: ui != jnp.inf, u)
    l_inf = tree_map(lambda li: li != -jnp.inf, l)
    upper = tree_where(u_inf, tree_sub(z, u), 0)  # mu in dual
    lower = tree_where(l_inf, tree_sub(l, z), 0)  # phi in dual
    return upper, lower
  
  return idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)


class OSQPState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    error: error used as stop criterion, deduced from residuals
    status: integer, one of ``[OSQP.UNSOLVED, OSQP.SOLVED, OSQP.PRIMAL_INFEASIBLE, OSQP.DUAL_INFEASIBLE]``.
    primal_residuals: residuals of constraints of primal problem
    dual_residuals: residuals of constraints of dual problem
    rho: per-constraint stepsizes, same pytree structure as (l,u) pair
    eq_qp_last_sol: solution of equality constrained QP. Useful for warm start.
  """
  iter_num: int
  error: float
  status: int
  primal_residuals: Any
  dual_residuals: Any
  rho: Any
  eq_qp_last_sol: Tuple[Any, Any]


def _make_matvec(matvec, params):
  if matvec is None:
    return lambda x: tree_map(jnp.dot, params, x)
  return lambda x: matvec(params, x)


@dataclass
class OSQP(base.IterativeSolver):
  """Operator Splitting Solver for Quadratic Programs.

  Jax implementation of the celebrated OSQP [1] based on ADMM.

  It solves convex problems of the form::
  
    \begin{aligned}
      \min_{x,z} \quad & \frac{1}{2}xPx + q^Tx\\
      \textrm{s.t.} \quad & Ax=z\\
        &l\leq z\leq u    \\
    \end{aligned}
  
  Equality constraints are obtained by setting l = u.
  If the inequality is one-sided then ``jnp.inf can be used for u,
  and ``-jnp.inf`` for l.

  P must be a positive semidefinite (PSD) matrix.

  The Lagrangian is given by::

    \mathcal{L} = 0.5x^TPx + q^Tx + y^T(Ax-z) + mu^T (z-u) + phi^T (l-z)

  Primal variables: x, z
  Dual variables  : y, mu, phi

  ADMM computes y at each iteration. mu and phi can be deduced from z and y.
  Defaults values for hyper-parameters come from: https://github.com/osqp/osqp/blob/master/include/constants.h

  Attributes:
    matvec_P: a Callable matvec_P(params_P, x).
      By default, matvec_P(P, x) = dot(P, x), where P = params_P.
    matvec_A: a Callable matvec_A(params_A, x).
      By default, matvec_A(A, x) = dot(A, x), where A = params_A.
    check_primal_dual_infeasability: if True populates the ``status`` field of ``state``
      with one of ``[OSQP.UNSOLVED, OSQP.SOLVED, OSQP.PRIMAL_INFEASIBLE, OSQP.DUAL_INFEASIBLE]``.
      If False decreases runtime (default: False).
    rho_start: initial stepsize for dual variables (default: 0.1 like paper)
    rho_active: multiplicator in stepsize for equality constraints
    sigma: ridge regularization parameter in linear system
    alpha: relaxation parameter (default: 1.6), must belong to open interval (0,2).
      alpha=1 => no relaxation.
      alpha<1 => under-relaxation
      alpha>1 => over-relaxation
      Boyd [2, p21] suggests chosing alpha in [1.5, 1.8]. 
    eq_qp_solve: linear solver for the equality constrained QP in ADMM iteration.
      (default: jaxopt.linear_solve.solve_normal_cg)
    eq_qp_solve_tol: tolerance for linear solver in equality constrained QP. (default: 1e-5)
      High tolerance may speedup each ADMM step but will slow down overall convergence. 
    eq_qp_solve_maxiter: number of iterations for linear solver in equality constrained QP. (default: None)
      Low maxiter will speedup each ADMM step but may slow down overall convergence.
    rho_start: initial learning rate. (default: 1e-1)
    rho_eq_over_rho_ineq: ratio between learning rates for equality and inequality constraints.
      Ratio bigger than 1 is recommended. (default: 1e3)
    rho_min: minimum learning rate. (default: 1e-6)
    rho_max: maximum learning rate. (default: 1e6)
    rho_tol: if ``u-l<rho_tol`` the corresponding dual variable
      benefits from the learning rate of equality constraints. (default: 1e-4)
    tol_primal_unfeasible: relative tolerance for primal infeasability detection. (default: 1e-4)
    tol_dual_unfeasible: relative tolerance for dual infeasability detection. (default: 1e-4)
    maxiter: maximum number of iterations.  (default: 4000)
    tol: absolute tolerance for stoping criterion (default: 1e-3).
    verbose: If verbose=1, print error at each iteration. If verbose=2, also print stepsizes and primal/dual variables.
      Warning: verbose>0 will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")

  [1] Stellato, B., Banjac, G., Goulart, P., Bemporad, A. and Boyd, S., 2020.
  OSQP: An operator splitting solver for quadratic programs.
  Mathematical Programming Computation, 12(4), pp.637-672.

  [2] Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J., 2010.
  Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.
  Machine Learning, 3(1), pp.1-122.
  """
  matvec_P: Optional[Callable] = None
  matvec_A: Optional[Callable] = None
  check_primal_dual_infeasability: bool = False
  sigma: float = 1e-6
  alpha: float = 1.6
  eq_qp_solve: Callable = solve_normal_cg
  eq_qp_solve_tol: Optional[float] = 1e-7  # relative tolerance. TODO: should it be absolute tolerance ?
  eq_qp_solve_maxiter: Optional[int] = None
  rho_start: float = 0.1
  rho_eq_over_rho_ineq: float = 1e3
  rho_min: float = 1e-6
  rho_max: float = 1e6
  rho_tol: float = 1e-4
  tol_primal_unfeasible: float = 1e-4  # relative tolerance
  tol_dual_unfeasible: float = 1e-4  # relative tolerance
  maxiter: int = 4000
  tol: float = 1e-3  # absolute tolerance. TODO: should it be absolute tolerance ?
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"


  # class attributes (ignored by @dataclass)
  UNSOLVED          = 0  # stopping criterion not reached yet
  SOLVED            = 1  # feasible solution found with satisfying precision
  DUAL_INFEASIBLE   = 2  # infeasible dual
  PRIMAL_INFEASIBLE = 3  # infeasible primal
  

  def _eq_constraints(self, box):
    l, u = box
    is_eq = lambda li, ui: (ui-li) < self.rho_tol
    return tree_map(is_eq, l, u)

  def init_state(self, init_params, params_obj, params_eq, params_ineq):
    x, z = init_params.primal
    y    = init_params.dual_eq  # good approximation for nu in Equality constrained QP
    rho  = tree_scalar_mul(self.rho_start, tree_ones_like(z))
    coef = tree_where(self._eq_constraints(params_ineq), self.rho_eq_over_rho_ineq, 1)
    rho  = tree_mul(coef, rho)
    return OSQPState(iter_num=0,
                     error=jnp.inf,
                     status=OSQP.UNSOLVED,
                     primal_residuals=jnp.inf,
                     dual_residuals=jnp.inf,
                     rho=rho,
                     eq_qp_last_sol=(x, y))
  
  def init_params(self, params_obj, params_eq, params_ineq, x0=None):
    """Return defaults params for initialization."""
    if x0 is None:
      x0 = tree_zeros_like(params_obj[1])  # like q
    z0 = self.matvec_A(params_eq)(x0)
    y0 = tree_zeros_like(z0)
    return base.KKTSolution((x0, z0), y0, (y0, y0))

  def _make_full_KKT_solution(self, primal, y):
    """Returns all dual variables of the problem."""
    # Unfortunately OSQP algorithm only returns y as dual variable,
    # mu and phi are missing, but can be recovered:
    #
    # We distinguish between l=u and l<u.
    # If l<u there are three cases:
    #   1. l < z < u: phi=0  mu=0 (and y=0)
    #   2. l = z < u: phi=-y mu=0 (and y<0)
    #   3. l < z = u: phi=0  mu=y (and y>0)
    #  this can be simplified with mu=relu(y) and phi=relu(-y)
    # If l=u then y=mu-phi then we have one degree of liberty to chose mu and phi
    # by symmetry with previous case we may chose mu=relu(y) and phi=relu(-y).
    is_pos = tree_map(lambda yi: yi >= 0, y)
    mu  = tree_where(is_pos, y, 0)  # derivative = 1 in y = 0
    phi = tree_map(lambda yi: jax.nn.relu(-yi), y)  # derivative = 0 in y = 0
    # y = mu - phi
    # d_y = d_mu - d_phi = 1 (everywhere; including in zero)
    return base.KKTSolution(primal=primal, dual_eq=y, dual_ineq=(mu, phi))

  def _update_stepsize(self, rho, primal_residuals, dual_residuals, P, q, A, box, x, y):
    """Update stepsize based on the ratio between primal and dual residuals."""
    # This heuristic was proven to be useful.
    # OSQP authors perform it on rescaled variables for better pre-conditioning.
    # This implementation does not use pre-conditioning, we keep their heuristic anyway.
    Ax, ATy     = _matvec_and_rmatvec(A, x, y)
    primal_coef = tree_norm_inf(primal_residuals) / tree_norm_inf(Ax)
    max_inf     = jnp.maximum(tree_norm_inf(P(x)), jnp.maximum(tree_norm_inf(ATy), tree_norm_inf(q)))
    dual_coef   = tree_norm_inf(dual_residuals) / max_inf
    coef = jnp.sqrt(primal_coef / (dual_coef + self.rho_min))
    rho_min = tree_scalar_mul(self.rho_min, tree_where(self._eq_constraints(box), self.rho_eq_over_rho_ineq, 1.))
    rho_max = tree_scalar_mul(self.rho_max, tree_where(self._eq_constraints(box), self.rho_eq_over_rho_ineq, 1.))
    rho = tree_map(jnp.clip, tree_scalar_mul(coef, rho), rho_min, rho_max)
    return rho

  def _compute_residuals(self, P, q, A, x, z, y):
    """Compute residuals of constraints for primal and dual, as defined in paper."""
    Ax, ATy = _matvec_and_rmatvec(A, x, y)
    primal_residuals = tree_sub(Ax, z)
    dual_residuals = tree_add(tree_add(P(x), q), ATy)
    return primal_residuals, dual_residuals

  def _compute_error(self, primal_residuals, dual_residuals, status):
    """Return error based on primal/dual residuals."""
    primal_res_inf = tree_norm_inf(primal_residuals)
    dual_res_inf = tree_norm_inf(dual_residuals)
    criterion = jnp.maximum(primal_res_inf, dual_res_inf)
    status = criterion <= self.tol
    status = jnp.where(status, OSQP.SOLVED, status)
    return criterion, status

  def _check_dual_infeasability(self, error, status, delta_x, P, q, Adx, l, u):
    """Check dual infeasability."""
    criterion  = self.tol_dual_unfeasible * tree_norm_inf(delta_x)

    certif_P   = tree_norm_inf(P(delta_x))
    certif_q   = tree_vdot(q, delta_x)

    unbouned_l = tree_map(lambda li: li == -jnp.inf, l)
    unbouned_u = tree_map(lambda ui: ui == jnp.inf, u)
    certif_l   = tree_map(lambda adxi,li: jnp.all(li <= adxi), Adx, tree_where(unbouned_l, -jnp.inf, -criterion))
    certif_u   = tree_map(lambda adxi,ui: jnp.all(adxi <= ui), Adx, tree_where(unbouned_u, jnp.inf, criterion))
    certif_A   = tree_reduce(jnp.logical_and, tree_map(jnp.logical_and, certif_l, certif_u))

    certif_dual_infeasible = jnp.logical_and(jnp.logical_and(certif_P <= criterion, certif_q <= criterion), certif_A)

    if self.verbose >= 2:
      print(f"certif_P={certif_P} certif_q={certif_q} certif_A={certif_A} criterion={criterion}, Adx={Adx}, certif_l={certif_l}, certif_u={certif_u}")

    # infeasible dual implies either infeasible primal, either unbounded primal.
    return jax.lax.cond(certif_dual_infeasible,
      lambda _: (0., OSQP.DUAL_INFEASIBLE),  # dual unfeasible; exit the main loop with error = 0.
      lambda _: (error, status),
      operand=None)

  def _check_primal_infeasability(self, error, status, delta_y, ATdy, l, u):
    """Check primal infeasability."""
    criterion = self.tol_primal_unfeasible * tree_norm_inf(delta_y)
    certif_A  = tree_norm_inf(ATdy)
    bounded_l = tree_where(tree_map(lambda li: li == -jnp.inf, l), 0., l)  # replace inf bounds by zero
    bounded_u = tree_where(tree_map(lambda ui: ui == jnp.inf, u), 0., u)
    dy_plus   = tree_map(jax.nn.relu, delta_y)
    dy_minus  = tree_neg(tree_map(jax.nn.relu, tree_neg(delta_y)))
    certif_lu = tree_add(tree_vdot(bounded_l, dy_minus), tree_vdot(bounded_u, dy_plus))
    certif_primal_infeasible = jnp.logical_and(certif_A  <= criterion, certif_lu  <= criterion)

    if self.verbose >= 2:
      print(f"certif_A={certif_A}, certif_lu={certif_lu}, criterion={criterion}")

    return jax.lax.cond(certif_primal_infeasible,
      lambda _: (0.,  # primal unfeasible; exit the main loop with error = 0.
                OSQP.PRIMAL_INFEASIBLE),  
      lambda _: (error, status),  # primal feasible or unbounded (depends of dual feasability).
      operand=None)  

  def _check_infeasability(self, prev_sol, sol, error, status, P, q, A, l, u):
    """Check primal and dual infeasability."""
    if not self.check_primal_dual_infeasability:
      return error, status

    delta_x = tree_sub(sol.primal[0], prev_sol.primal[0])
    delta_y = tree_sub(sol.dual_eq, prev_sol.dual_eq)
    Adx, ATdy = _matvec_and_rmatvec(A, delta_x, delta_y)

    error, status = self._check_dual_infeasability(error, status, delta_x, P, q, Adx, l, u)
    error, status = self._check_primal_infeasability(error, status, delta_y, ATdy, l, u)

    return error, status

  def _solve_linear_system(self, params, P, q, A, inv_rho, eq_qp_last_sol):
    """ Solve equality constrained QP in ADMM split."""
    # solve the "augmented" equality constrained QP:
    #
    #     minimize 0.5x_bar P x_bar + q x_bar 
    #     (1)        + (sigma/2) \|x_bar - x\|^2_2
    #     (2)        + (rho/2)   \|z_bar - z + rho^{-1} y\|^2_2
    #     under    A x_bar = z_bar; x_bar = x
    #
    #        (1) and (2) come from the augmented Lagrangian
    #
    # This problem is easy to solve by writing the KKT optimality conditions.
    # By construction the solution is unique without imposing strict convexity of objective nor 
    # independance of the constraints.
    # The primal feasability conditions are used to eliminate z_bar from the system (which simplifies it).
    # The stationarity conditons obtained by derivating the Lagrangian wrt x_bar and z_bar
    # constitute the two parts of this system.
    x, z = params.primal
    y    = params.dual_eq  # dual variables for constraints z_bar = z;
    def matvec(unknowns):
      x_bar, nu = unknowns
      Ax_bar, AT_nu = _matvec_and_rmatvec(A, x_bar, nu)
      stationarity_x_bar = tree_add(tree_add_scalar_mul(P(x_bar), self.sigma, x_bar), AT_nu)
      stationarity_z_bar = tree_add(Ax_bar, tree_mul(tree_neg(inv_rho), nu))
      return stationarity_x_bar, stationarity_z_bar
    b1 = tree_sub(tree_scalar_mul(self.sigma, x), q)
    b2 = tree_sub(z, tree_mul(inv_rho, y))
    x_bar, nu = self.eq_qp_solve(matvec, (b1, b2), x0=eq_qp_last_sol,
                                 maxiter=self.eq_qp_solve_maxiter,
                                 tol=self.eq_qp_solve_tol)
    return x_bar, nu

  def _admm_step(self, params, P, q, A, box, rho, eq_qp_last_sol):
    """Perform atomic step of ADMM algorithm."""
    x, z = params.primal
    y    = params.dual_eq  # dual variables for constraints z_bar = z;
    # mu, phi = params.dual_ineq are unused

    inv_rho = tree_map(lambda r: 1/r, rho)

    # lines are numbered according to the pseudo-code in the paper OSQP.

    # line 3: optimization step for (x_bar, z_bar)
    # this equality constrained QP is solved by writing KKT conditions
    # which fall back to a well posed linear system
    x_bar, nu = self._solve_linear_system(params, P, q, A, inv_rho, eq_qp_last_sol)
    z_bar = tree_add(z, tree_mul(inv_rho, tree_sub(nu, y)))  # line 4

    # line 5: optimization step for x with relaxation parameter alpha (smooth updates)
    # alpha can be understood as momentum term
    x_next = tree_add(x, tree_scalar_mul(self.alpha, tree_sub(x_bar, x)))

    # line 6: optimization step for z with relaxation parameter alpha (smooth updates)
    # by definition A x_bar = z_bar and l <= z <= u thanks to projection
    # the dual variable y corresponds to constraint z_bar = z
    z_momentum = tree_add(z, tree_scalar_mul(self.alpha, tree_sub(z_bar, z)))
    z_step = tree_mul(inv_rho, y)
    z_free = tree_add(z_momentum, z_step)
    z_next = projection_box(z_free, box)

    # line 7: gradient descent on dual variables, with relaxation
    y_step = tree_sub(z_momentum, z_next)
    y_next = tree_add(y, tree_mul(rho, y_step))

    return (x_next, z_next), y_next, (x_bar, nu)

  def update(self, params, state, params_obj, params_eq, params_ineq):
    """Perform ADMM step and update stepsizes, compute dual variables, residuals."""
    # The original problem on variables (x,z) is split into TWO problems
    # with variables (x, z) and (x_bar, z_bar)
    # 
    # (x_bar, z_bar) is NOT part of the state because it is recomputed at each step:
    #    (x_bar, z_bar) = argmin_{x_bar, z_bar} L(x_bar, z_bar, x, z, y)
    # with L the augmented Lagrangian
    # z_bar is always such that A x_bar = z_bar
    #
    # x = argmin_x L(x_bar, z_bar, x, z, y)
    # for equality constraint x = x_bar the dual variable is constant (=0) and can be eliminated
    #
    # z = argmin_z L(x_bar, z_bar, x, z, y)
    # for equality constraint z = z_bar the associated dual variable is y
    P    = self.matvec_P(params_obj[0])
    q    = params_obj[1]
    A    = self.matvec_A(params_eq)
    l, u = params_ineq

    # for active constraints (in particular equality constraints) high stepsize is better
    rho = state.rho
    if self.verbose >= 2:
      rho_eq = tree_reduce(jnp.maximum, tree_map(jnp.max, rho))
      rho_ineq = tree_reduce(jnp.minimum, tree_map(jnp.min, rho))
      print(f"rho_eq={rho_eq} rho_ineq={rho_ineq}")

    (x, z), y, eq_qp_last_sol = self._admm_step(params, P, q, A, (l, u), rho, state.eq_qp_last_sol)
    if self.verbose >= 3:
      print(f"x={x}\nz={z}\ny={y}")

    primal_residuals, dual_residuals = self._compute_residuals(P, q, A, x, z, y)
    if self.verbose >= 3:
      print(f"primal_residuals={primal_residuals}, dual_residuals={dual_residuals}")

    rho = self._update_stepsize(state.rho, primal_residuals, dual_residuals, P, q, A, (l, u), x, y)

    sol = self._make_full_KKT_solution(primal=(x, z), y=y)
    error, status = self._compute_error(primal_residuals, dual_residuals, state.status)
    error, status = self._check_infeasability(params, sol, error, status, P, q, A, l, u)      

    state = OSQPState(iter_num=state.iter_num+1,
                      error=error,
                      status=status,
                      primal_residuals=primal_residuals,
                      dual_residuals=dual_residuals,
                      rho=rho,
                      eq_qp_last_sol=eq_qp_last_sol)
    return base.OptStep(params=sol, state=state)

  def l2_optimality_error(
      self,
      params: base.KKTSolution,
      params_obj: Tuple[Any, Any],
      params_eq: Any,
      params_ineq: Tuple[Any, Any]) -> base.OptStep:
    """Computes the L2 norm of the KKT residuals."""
    pytree = self.optimality_fun(params, params_obj, params_eq, params_ineq)
    return tree_l2_norm(pytree)

  def __post_init__(self):
    # If no user-defined function is provided 
    self.matvec_P = partial(_make_matvec, self.matvec_P)
    self.matvec_A = partial(_make_matvec, self.matvec_A)

    self.optimality_fun = _make_osqp_optimality_fun(self.matvec_P, self.matvec_A)
