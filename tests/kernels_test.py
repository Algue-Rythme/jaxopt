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

"""Kernels tests."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as onp
from numpy.testing import assert_array_almost_equal

from jaxopt._src.kernels import LinearKernel, RBFKernel, LaplacianKernel, NystromKernel
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn import datasets
from sklearn.kernel_approximation import Nystroem


class KernelsTest(jtu.JaxTestCase):

  def test_linear_kernel(self):
    onp.random.seed(13)
    num_samples_X = 10
    num_samples_Y = 8
    num_features = 5
    X = onp.random.randn(num_samples_X, num_features)
    Y = onp.random.randn(num_samples_Y, num_features)
    K = X @ Y.T
    kernel = LinearKernel()
    params = kernel.init_params(X, Y)
    dummy = jnp.zeros(num_samples_Y)
    K_dense = kernel.get_dense(params, dummy)
    self.assertArraysAllClose(K, K_dense, rtol=1e-5, atol=1e-5)

  def test_rbf_skln(self):
    num_samples = 20
    num_features = 6
    gamma = 1. / num_features
    data, _ = datasets.make_regression(n_samples=num_samples,
                                       n_features=num_features,
                                       random_state=0)
    kernel = RBFKernel(gamma=gamma)
    kernel_params = kernel.init_params(data)

    # test with random coefficients
    vec = onp.random.rand(num_samples)
    y = kernel.matvec(kernel_params, vec)
    y_skln = rbf_kernel(data, gamma=gamma) @ vec
    self.assertArraysAllClose(y, y_skln, rtol=1e-5, atol=1e-5)

  def test_laplacian_skln(self):
    num_samples = 20
    num_features = 6
    gamma = 1. / num_features
    data, _ = datasets.make_regression(n_samples=num_samples,
                                       n_features=num_features,
                                       random_state=0)
    kernel = LaplacianKernel(gamma=gamma)
    kernel_params = kernel.init_params(data)

    # test with random coefficients
    vec = onp.random.rand(num_samples)
    y = kernel.matvec(kernel_params, vec)
    y_skln = laplacian_kernel(data, gamma=gamma) @ vec
    self.assertArraysAllClose(y, y_skln, rtol=1e-5, atol=1e-5)

  @parameterized.product(rank=[10, 20, 50, 100, 200, 500, 800, 1000])
  def test_nystrom(self, rank):
    onp.random.seed(66)
    num_samples = 1000
    num_features = 7
    data = onp.random.randn(num_samples, num_features)

    gamma = 1 / num_features

    nystrom_skln = Nystroem(kernel='rbf', gamma=gamma, n_components=rank, random_state=0)
    feature_map = nystrom_skln.fit_transform(data)

    # Since Sklearn samples at random the landmark points (no user defined landmarks)
    # we must apply the same permutation in input/output of Jaxopt in order to check the results.
    comp_indices = nystrom_skln.component_indices_
    is_not_comp_index = onp.full(num_samples, fill_value=True)
    is_not_comp_index[comp_indices] = False
    other_indices = onp.array([i for i in range(num_samples) if is_not_comp_index[i]], dtype=onp.int64)
    def perm(mat_or_vec):
      if len(mat_or_vec.shape) > 1:
        a = mat_or_vec[comp_indices,:]
        b = mat_or_vec[other_indices,:]
      else:
        a = mat_or_vec[comp_indices]
        b = mat_or_vec[other_indices]
      return jnp.concatenate([a, b], axis=0)

    perm_data = perm(data)

    rbf = RBFKernel(gamma=gamma)
    rbf_params_perm = rbf.init_params(perm_data)
    rbf_params = rbf.init_params(data)

    nystrom = NystromKernel(kernel=rbf, rank=rank)
    nystrom_params_perm = nystrom.init_params(perm_data)

    vec = onp.random.rand(num_samples)
    perm_vec = perm(vec)

    y_nystrom = nystrom.matvec(nystrom_params_perm, perm_vec)
    y_rbf_perm = rbf.matvec(rbf_params_perm, perm_vec)
    y_rbf = perm(rbf.matvec(rbf_params, vec))
    y_rbf_skln = perm(rbf_kernel(data, gamma=gamma) @ vec)
    y_nys_skln = perm(jnp.dot(feature_map, jnp.dot(feature_map.T, vec)))

    # Sanity check: RBF kernel with/without permutations should match,
    # both in jaxopt and sklearn implementations.
    self.assertArraysAllClose(y_rbf_perm, y_rbf, rtol=1e-5, atol=1e-5)
    self.assertArraysAllClose(y_rbf_perm, y_rbf_skln, rtol=1e-5, atol=1e-5)

    atol = 1e-4
    rtol = 1e-2

    # Consistency check: Nystrom approximations of Jaxopt and Sklearn must agree.
    if jax.config.read("jax_enable_x64"):
      # coefficientwise in float64
      self.assertArraysAllClose(y_nystrom, y_nys_skln, rtol=rtol, atol=atol)
    else:
      # relative error in float32.
      rel_err = jnp.linalg.norm(y_nystrom - y_nys_skln) / jnp.linalg.norm(y_nys_skln)
      self.assertAllClose(rel_err, 0., atol=rtol)
    
    if rank == num_samples:
      # Approximation check: Nystrom approximation and true RBF kernel should be close.
      self.assertArraysAllClose(y_nystrom, y_rbf_perm, rtol=rtol, atol=atol)
      self.assertArraysAllClose(y_nys_skln, y_rbf_perm, rtol=rtol, atol=atol)
    
    else:

      nys_rbf_err = jnp.linalg.norm(y_nystrom - y_rbf_perm)
      skln_rbf_err = jnp.linalg.norm(y_rbf_perm - y_nys_skln)
      if jax.config.read("jax_enable_x64"):
        # Similar error between Jaxopt and Sklearn wrt ground truth RBF.
        self.assertAllClose(nys_rbf_err, skln_rbf_err, rtol=rtol, atol=atol)
      else:
        # Error of Jaxopt in 32 bits can be three times higher than Sklearn in 64 bits.
        self.assertAllClose(nys_rbf_err, skln_rbf_err, rtol=3.)

      # Check that the submatrix of shape (rank,rank) is correctly recovered.
      # This is guaranteed mathematically and should behaves correctly numerically.
      y_nys_truncated = y_nystrom[:rank]
      y_rbf_truncated = y_rbf_perm[:rank]
      self.assertAllClose(y_nys_truncated, y_rbf_truncated, rtol=rtol, atol=atol)
      
      # Compare error on kernels.
      # Note that Sklearn does not perform this test when rank < num_samples:
      # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tests/test_kernel_approximation.py
      # Indeed when rank << num_samples the error elementwise can be quite high.
      K = rbf.get_dense(rbf_params_perm, perm_vec)
      K_nys = nystrom.get_dense(nystrom_params_perm, perm_vec)

      # We use the relative error instead.
      rel_err = jnp.linalg.norm(K.ravel() - K_nys.ravel()) / jnp.linalg.norm(K.ravel())
      # We set the tolerance as decreasing function of the rank.
      err_ratio = jnp.maximum(1e-2, 10. / rank)
      self.assertAllClose(rel_err, 0., atol=err_ratio)


if __name__ == '__main__':
  jax.config.update("jax_enable_x64", False)
  absltest.main(testLoader=jtu.JaxTestLoader())
