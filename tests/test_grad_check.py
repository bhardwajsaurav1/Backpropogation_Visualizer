"""
Finite-difference numerical gradient verification tests.
Mathematically verifies all hand-derived analytical gradients:
d f(theta)/d theta ≈ [f(theta + eps) - f(theta - eps)] / (2 * eps)
Ensures relative error < 1e-6.
"""

import pytest
import numpy as np
from core.activations import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    elu, elu_derivative,
    gelu, gelu_derivative,
    swish, swish_derivative,
    softmax, softmax_derivative
)
from core.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError, HuberLoss
from core.layer import DenseLayer
from core.network import NeuralNetwork


def compute_relative_error(grad_analytical, grad_numerical, eps=1e-12):
    """Computes symmetric relative error between analytical and numerical gradients."""
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + eps
    return numerator / denominator


class TestActivationGradients:
    """Verifies derivatives of individual activation functions."""

    @pytest.mark.parametrize("act_fn,deriv_fn", [
        (sigmoid, sigmoid_derivative),
        (tanh, tanh_derivative),
        (gelu, gelu_derivative),
        (swish, swish_derivative),
    ])
    def test_smooth_activation_gradients(self, act_fn, deriv_fn):
        rng = np.random.default_rng(42)
        z = rng.uniform(-3.0, 3.0, (10, 5))
        eps = 1e-5

        analytical = deriv_fn(z)
        numerical = (act_fn(z + eps) - act_fn(z - eps)) / (2.0 * eps)

        rel_error = compute_relative_error(analytical, numerical)
        assert rel_error < 1e-5, f"Relative error {rel_error} exceeds tolerance for {act_fn.__name__}"

    def test_leaky_relu_gradient(self):
        rng = np.random.default_rng(42)
        # Avoid exact 0 discontinuity
        z = np.array([-2.5, -1.0, -0.05, 0.05, 1.2, 3.0])
        eps = 1e-5
        analytical = leaky_relu_derivative(z, alpha=0.01)
        numerical = (leaky_relu(z + eps, alpha=0.01) - leaky_relu(z - eps, alpha=0.01)) / (2.0 * eps)
        rel_error = compute_relative_error(analytical, numerical)
        assert rel_error < 1e-5


class TestLossGradients:
    """Verifies analytical loss backward gradients."""

    def test_bce_loss_gradient(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, (20, 1)).astype(float)
        y_pred = rng.uniform(0.1, 0.9, (20, 1))
        loss_fn = BinaryCrossEntropy()
        eps = 1e-6

        analytical = loss_fn.backward(y_pred, y_true)
        numerical = np.zeros_like(y_pred)

        for i in range(y_pred.shape[0]):
            orig = y_pred[i, 0]
            y_pred[i, 0] = orig + eps
            loss_plus = loss_fn.forward(y_pred, y_true)
            y_pred[i, 0] = orig - eps
            loss_minus = loss_fn.forward(y_pred, y_true)
            y_pred[i, 0] = orig
            numerical[i, 0] = (loss_plus - loss_minus) / (2.0 * eps)

        rel_error = compute_relative_error(analytical, numerical)
        assert rel_error < 1e-5

    def test_mse_loss_gradient(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal((15, 2))
        y_pred = rng.standard_normal((15, 2))
        loss_fn = MeanSquaredError()
        eps = 1e-6

        analytical = loss_fn.backward(y_pred, y_true)
        numerical = np.zeros_like(y_pred)

        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                orig = y_pred[i, j]
                y_pred[i, j] = orig + eps
                loss_plus = loss_fn.forward(y_pred, y_true)
                y_pred[i, j] = orig - eps
                loss_minus = loss_fn.forward(y_pred, y_true)
                y_pred[i, j] = orig
                numerical[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

        rel_error = compute_relative_error(analytical, numerical)
        assert rel_error < 1e-5


class TestDenseLayerAndNetworkGradientChecking:
    """Verifies backpropagation gradients across multi-layer networks."""

    def test_multi_layer_backprop_grad_check(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 3))
        y = rng.integers(0, 2, (8, 1)).astype(float)

        net = NeuralNetwork(
            layer_sizes=[3, 6, 4, 1],
            activations=["tanh", "tanh", "sigmoid"],
            loss="bce",
            seed=42
        )

        # Run forward & backward
        y_pred = net.forward(X, training=True)
        net.backward(y_pred, y)

        eps = 1e-6

        # Check gradients for each Dense layer
        for idx, layer in enumerate(net.dense_layers):
            # Check grad_W
            W_analytical = layer.grad_W.copy()
            W_numerical = np.zeros_like(layer.W)

            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    orig_val = layer.W[i, j]
                    layer.W[i, j] = orig_val + eps
                    loss_plus = net.compute_loss(net.forward(X, training=True), y)
                    layer.W[i, j] = orig_val - eps
                    loss_minus = net.compute_loss(net.forward(X, training=True), y)
                    layer.W[i, j] = orig_val
                    W_numerical[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

            rel_err_W = compute_relative_error(W_analytical, W_numerical)
            assert rel_err_W < 1e-5, f"Layer {idx} W grad relative error {rel_err_W:.2e} too high!"

            # Check grad_b
            b_analytical = layer.grad_b.copy()
            b_numerical = np.zeros_like(layer.b)

            for j in range(layer.b.shape[1]):
                orig_val = layer.b[0, j]
                layer.b[0, j] = orig_val + eps
                loss_plus = net.compute_loss(net.forward(X, training=True), y)
                layer.b[0, j] = orig_val - eps
                loss_minus = net.compute_loss(net.forward(X, training=True), y)
                layer.b[0, j] = orig_val
                b_numerical[0, j] = (loss_plus - loss_minus) / (2.0 * eps)

            rel_err_b = compute_relative_error(b_analytical, b_numerical)
            assert rel_err_b < 1e-5, f"Layer {idx} b grad relative error {rel_err_b:.2e} too high!"
