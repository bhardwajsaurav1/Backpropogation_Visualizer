"""
Layer abstractions for Dense and Dropout layers with transparent forward/backward passes.
Computes analytical gradients for weights, biases, and inputs with full intermediate caching.
"""

import numpy as np
from .activations import get_activation


def initialize_weights(in_dim, out_dim, mode="he_normal", rng=None):
    """
    Weight initializations:
    - 'he_normal' / 'he_uniform': Kaiming init, optimal for ReLU/LeakyReLU
    - 'xavier_normal' / 'xavier_uniform': Glorot init, optimal for Sigmoid/Tanh
    - 'lecun': Optimal for SELU
    - 'normal': Gaussian with std=0.01
    - 'zero': Zero initialization (demonstrates symmetry breaking problem)
    """
    if rng is None:
        rng = np.random.default_rng()

    mode = mode.lower()
    if mode == "he_normal":
        std = np.sqrt(2.0 / in_dim)
        return rng.standard_normal((in_dim, out_dim)) * std
    elif mode == "he_uniform":
        limit = np.sqrt(6.0 / in_dim)
        return rng.uniform(-limit, limit, (in_dim, out_dim))
    elif mode == "xavier_normal" or mode == "glorot_normal":
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return rng.standard_normal((in_dim, out_dim)) * std
    elif mode == "xavier_uniform" or mode == "glorot_uniform":
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return rng.uniform(-limit, limit, (in_dim, out_dim))
    elif mode == "lecun":
        std = np.sqrt(1.0 / in_dim)
        return rng.standard_normal((in_dim, out_dim)) * std
    elif mode == "normal":
        return rng.standard_normal((in_dim, out_dim)) * 0.1
    elif mode == "zero":
        return np.zeros((in_dim, out_dim))
    elif mode == "constant":
        return np.ones((in_dim, out_dim)) * 0.5
    else:
        std = np.sqrt(1.0 / in_dim)
        return rng.standard_normal((in_dim, out_dim)) * std


class DenseLayer:
    """
    Fully connected (affine) layer with non-linear activation.
    Operations:
      Forward:  Z = X @ W + b
                A = activation(Z)
      Backward: delta = dL/dA * activation_derivative(Z)  [or direct dL/dZ]
                dL/dW = X.T @ delta
                dL/db = sum(delta, axis=0)
                dL/dX = delta @ W.T
    """
    def __init__(self, in_features, out_features, activation="relu",
                 init_mode="he_normal", l1=0.0, l2=0.0, rng=None, name="Dense"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.act_fn, self.act_deriv = get_activation(activation)
        self.init_mode = init_mode
        self.l1 = l1
        self.l2 = l2
        self.name = name

        # Initialize parameters
        self.W = initialize_weights(in_features, out_features, mode=init_mode, rng=rng)
        self.b = np.zeros((1, out_features))

        # Gradient storage
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Micro-inspection cache for forward and backward passes
        self.cache = {}

    def forward(self, X, training=True):
        """
        Forward pass through linear transform and non-linear activation.
        X shape: (batch_size, in_features)
        """
        Z = X @ self.W + self.b
        A = self.act_fn(Z)

        self.cache = {
            "X": X,
            "Z": Z,
            "A": A,
            "training": training
        }
        return A

    def backward(self, upstream_grad, is_direct_dZ=False):
        """
        Backward pass computing parameter and input gradients.
        upstream_grad:
          - if is_direct_dZ is False: dL/dA (upstream gradient wrt post-activation A)
          - if is_direct_dZ is True: dL/dZ (pre-computed gradient wrt pre-activation Z, e.g. for output layer)
        Returns:
          dL/dX to pass to the preceding layer.
        """
        X = self.cache["X"]
        Z = self.cache["Z"]
        A = self.cache["A"]

        if is_direct_dZ:
            dZ = upstream_grad
        else:
            # Chain rule through activation: dL/dZ = dL/dA * act'(Z)
            dZ = upstream_grad * self.act_deriv(Z)

        # Gradients wrt parameters (matrix calculus)
        # dL/dW = X^T @ dZ
        self.grad_W = X.T @ dZ
        # dL/db = sum along batch axis
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # Add regularization gradients
        if self.l2 > 0.0:
            self.grad_W += self.l2 * self.W
        if self.l1 > 0.0:
            self.grad_W += self.l1 * np.sign(self.W)

        # Gradient wrt input to propagate upstream: dL/dX = dZ @ W^T
        dX = dZ @ self.W.T

        # Store delta and dZ for visualization and diagnostics
        self.cache["dZ"] = dZ
        self.cache["upstream_grad"] = upstream_grad
        self.cache["dX"] = dX

        return dX

    def get_regularization_loss(self):
        """Computes L1 and L2 penalty for current weights."""
        reg_loss = 0.0
        if self.l2 > 0.0:
            reg_loss += 0.5 * self.l2 * np.sum(self.W ** 2)
        if self.l1 > 0.0:
            reg_loss += self.l1 * np.sum(np.abs(self.W))
        return reg_loss


class DropoutLayer:
    """
    Inverted Dropout regularization layer.
    Scales remaining activations by 1/(1-p) during training so inference requires no scaling.
    """
    def __init__(self, p=0.2, rng=None):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mask = None
        self.cache = {}

    def forward(self, X, training=True):
        if training and self.p > 0.0:
            # Keep probability
            keep_prob = 1.0 - self.p
            self.mask = (self.rng.uniform(0.0, 1.0, size=X.shape) < keep_prob) / keep_prob
            out = X * self.mask
        else:
            self.mask = None
            out = X

        self.cache = {"X": X, "training": training, "mask": self.mask}
        return out

    def backward(self, upstream_grad):
        if self.mask is not None:
            return upstream_grad * self.mask
        return upstream_grad

    def get_regularization_loss(self):
        return 0.0
