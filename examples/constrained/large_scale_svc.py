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
Large scale SVC with intercept and Nystrom approximation.
=========================================================

One versus all classification on Mnist.
"""
from absl import app
from absl import flags

import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt import BoxOSQP

import numpy as onp
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm

import time

import tensorflow as tf
import numpy as onp


flags.DEFINE_float("lam", 0.5, "Regularization parameter. Must be positive.")
flags.DEFINE_float("tol", 1e-6, "Tolerance of solvers.")
flags.DEFINE_integer("dataset_size", 1000, "Run on subset of train set.")
flags.DEFINE_integer("rank", 1000, "Rank of Nystrom approximation.")
flags.DEFINE_bool("verbose", False, "Verbosity.")
FLAGS = flags.FLAGS


@dataclass
class SVC_OSQP:
  lam: float
  C: float
  kernel: Kernel
  tol: float = 1e-1
  verbose: int = 0

  def train_one_versus_all(self, kernel_params, labels, label):
    # The dual objective is:
    # fun(beta) = 0.5 beta^T K beta - beta^T y
    # subject to
    # sum(beta) = 0
    # 0 <= beta_i <= C if y_i = 1
    # -C <= beta_i <= 0 if y_i = -1
    # where C = 1.0 / lam

    print("Train One versus All on label {label}.")

    y = labels == label
    y = y * 2. - 1.

    def matvec_Q(kernel_params, beta):
      return 0.5 * self.kernel.matvec(kernel_params, beta)

    def matvec_A(_, beta):
      return beta, jnp.sum(beta)

    l = -jax.nn.relu(-y * self.C), 0.
    u =  jax.nn.relu( y * self.C), 0.

    hyper_params = dict(params_obj=(kernel_params, -y), params_eq=None, params_ineq=(l, u))
    osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A,
                   tol=self.tol, maxiter=500, verbose=self.verbose)

    if self.verbose == 0:
      beta = osqp.run(init_params=None, **hyper_params)
    else:

      @jax.jit
      def jitted_update(params, state):
        return osqp.update(params, state, **hyper_params)

      params = osqp.init_params(init_x=None, **hyper_params)
      state = osqp.init_state(init_params=params, **hyper_params)

      for iter_num in range(osqp.maxiter):
        params, state = jitted_update(params, state)
        print(f"[{iter_num}] error={state.error:.5f}")
        if state.error < self.tol:
          break

      beta = params.primal[0]

    print(f"Label={label} Final error: {state.error:.7f}")

    return beta * y_train

  def train_one_versus_all(data, labels):
    # Independant of labels.
    start = time.time()
    kernel_params = self.kernel.init_params(data)

    coefficients = {}
    for label in onp.unique(labels):
      coefs = self.train_one_versus_all(kernel_params, labels, label)
      coefficients[int(label)] = coefs

    durations = time.time() - start
    print(f"Duration   : {duarations:.2f} seconds")

    return kernel_params, coefficients

  def predict(self, svm_params, test_set):
    predict_params = self.kernel.get_predict_params(kernel_params, beta_fit_osqp)
    y_pred = self.kernel.predict(predict_params, x_test)


def plot_support_vectors(x_train, alpha):
  support = alpha > 1e-5
  indices = onp.where(support)[0]
  num_print = 16
  n_col = 4
  n_row = len(indices[:num_print]) // n_col
  _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
  axs = axs.flatten()
  imgs = [x_train[i].reshape((28,28)) for i in indices[:num_print]]
  for img, ax in zip(imgs, axs):
      ax.imshow(img)
  plt.show()


def main(argv):
  del argv

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
  x_train = x_train.reshape((60*1000, 28*28))
  x_train = x_train / 256
  x_test = x_test / 256
  x_test = x_test.reshape((10*1000, 28*28))

  # Shuffling required to make Nystrom work.
  onp.random.seed(67)
  indices = onp.random.permutation(len(X))
  x_train = x_train[indices,:]
  y = y[indices]

  # Select subset.
  x_train = x_train[:dataset_size,:]
  y_train = y_train[:dataset_size]

  gamma = 1.0 / (784 * X.var())
  rbf = RBFKernel(gamma)
  rank = FLAGS.rank
  kernel = NystromKernel(rank=rank, kernel=rbf)

  svc = SVC_OSQP(kernel=kernel)
  svc.train()



if __name__ == "__main__":
  # jax.config.update("jax_platform_name", "cpu")
  app.run(main)
