"""
Activation functions and their exact hand-derived derivatives.
All implementations are pure NumPy, vectorized, and numerically stabilized.
"""

import numpy as np


def sigmoid(z):
    """
    Sigmoid activation: σ(z) = 1 / (1 + e^(-z))
    Clips z to [-500, 500] to prevent floating point overflow in exp.
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(z_or_a, is_post_activation=False):
    """
    dσ/dz = σ(z) * (1 - σ(z))
    If is_post_activation is True, input is a = σ(z).
    """
    if is_post_activation:
        a = z_or_a
    else:
        a = sigmoid(z_or_a)
    return a * (1.0 - a)


def tanh(z):
    """
    Hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    """
    return np.tanh(z)


def tanh_derivative(z_or_a, is_post_activation=False):
    """
    d tanh(z) / dz = 1 - tanh^2(z)
    If is_post_activation is True, input is a = tanh(z).
    """
    if is_post_activation:
        a = z_or_a
    else:
        a = tanh(z_or_a)
    return 1.0 - a ** 2


def relu(z):
    """
    Rectified Linear Unit: ReLU(z) = max(0, z)
    """
    return np.maximum(0.0, z)


def relu_derivative(z_or_a, is_post_activation=False):
    """
    d ReLU(z) / dz = 1 if z > 0 else 0
    """
    return (z_or_a > 0.0).astype(float)


def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU: LeakyReLU(z) = z if z > 0 else alpha * z
    """
    return np.where(z > 0.0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    """
    d LeakyReLU(z) / dz = 1 if z > 0 else alpha
    """
    return np.where(z > 0.0, 1.0, alpha)


def elu(z, alpha=1.0):
    """
    Exponential Linear Unit: ELU(z) = z if z > 0 else alpha * (e^z - 1)
    """
    z_clipped = np.clip(z, -500, 500)
    return np.where(z > 0.0, z, alpha * (np.exp(z_clipped) - 1.0))


def elu_derivative(z_or_a, alpha=1.0, is_post_activation=False):
    """
    d ELU(z) / dz = 1 if z > 0 else ELU(z) + alpha
    """
    if is_post_activation:
        a = z_or_a
        return np.where(a > 0.0, 1.0, a + alpha)
    else:
        z = z_or_a
        return np.where(z > 0.0, 1.0, alpha * np.exp(np.clip(z, -500, 500)))


def gelu(z):
    """
    Gaussian Error Linear Unit (Approximation):
    GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/π) * (z + 0.044715 * z^3)))
    """
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))


def gelu_derivative(z):
    """
    Exact derivative of the tanh approximation of GELU.
    """
    k = np.sqrt(2.0 / np.pi)
    inner = k * (z + 0.044715 * (z ** 3))
    tanh_inner = np.tanh(inner)
    sech2_inner = 1.0 - tanh_inner ** 2
    d_inner = k * (1.0 + 3.0 * 0.044715 * (z ** 2))
    return 0.5 * (1.0 + tanh_inner) + 0.5 * z * sech2_inner * d_inner


def swish(z, beta=1.0):
    """
    Swish / SiLU activation: Swish(z) = z * σ(β * z)
    """
    s = sigmoid(beta * z)
    return z * s


def swish_derivative(z, beta=1.0):
    """
    d Swish(z) / dz = β * Swish(z) + σ(β*z) * (1 - β * Swish(z))
    """
    s = sigmoid(beta * z)
    return beta * z * s * (1.0 - s) + s


def linear(z):
    """
    Linear (Identity) activation: f(z) = z
    """
    return z


def linear_derivative(z):
    """
    d Linear(z) / dz = 1
    """
    return np.ones_like(z, dtype=float)


def softmax(z):
    """
    Softmax activation along last axis (for multi-class classification).
    Numerically stabilized with shift-by-max: softmax(z_i) = e^(z_i - max(z)) / sum(e^(z_j - max(z)))
    """
    # Shift z for numerical stability
    shifted_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def softmax_derivative(a):
    """
    Softmax Jacobian per sample: J_ij = a_i * (δ_ij - a_j).
    When combined with Categorical Cross Entropy Loss, the combined gradient simplifies to:
    ∂L/∂z = a - y
    """
    # Returns diagonal a - a @ a.T for single vector or batch
    return a * (1.0 - a)


ACTIVATION_REGISTRY = {
    "sigmoid": {
        "fn": sigmoid,
        "derivative": sigmoid_derivative,
        "formula": r"\sigma(z) = \frac{1}{1 + e^{-z}}",
        "deriv_formula": r"\sigma'(z) = \sigma(z)(1 - \sigma(z))",
        "range": "(0, 1)",
        "description": "Smooth S-curve mapping inputs to probabilities (0, 1). Susceptible to vanishing gradients at extreme values."
    },
    "tanh": {
        "fn": tanh,
        "derivative": tanh_derivative,
        "formula": r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}",
        "deriv_formula": r"\tanh'(z) = 1 - \tanh^2(z)",
        "range": "(-1, 1)",
        "description": "Zero-centered S-curve. Stronger gradients than sigmoid, but still saturates at extremes."
    },
    "relu": {
        "fn": relu,
        "derivative": relu_derivative,
        "formula": r"\text{ReLU}(z) = \max(0, z)",
        "deriv_formula": r"\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}",
        "range": r"[0, \infty)",
        "description": "Standard deep learning activation. Fast to compute, avoids vanishing gradient for positive activations, but can cause dying neurons."
    },
    "leaky_relu": {
        "fn": leaky_relu,
        "derivative": leaky_relu_derivative,
        "formula": r"\text{LeakyReLU}(z) = \max(\alpha z, z)",
        "deriv_formula": r"\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \le 0 \end{cases}",
        "range": r"(-\infty, \infty)",
        "description": "Prevents dead neurons by introducing a small constant slope (alpha=0.01) for negative inputs."
    },
    "elu": {
        "fn": elu,
        "derivative": elu_derivative,
        "formula": r"\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases}",
        "deriv_formula": r"\text{ELU}'(z) = \begin{cases} 1 & z > 0 \\ \text{ELU}(z) + \alpha & z \le 0 \end{cases}",
        "range": r"(-\alpha, \infty)",
        "description": "Smooth non-zero gradient for negative inputs, bringing mean activations closer to zero."
    },
    "gelu": {
        "fn": gelu,
        "derivative": gelu_derivative,
        "formula": r"\text{GELU}(z) = z \cdot \Phi(z)",
        "deriv_formula": r"\text{GELU}'(z) \approx 0.5(1 + \tanh(\cdot)) + \dots",
        "range": r"(-0.17, \infty)",
        "description": "State-of-the-art activation used in Modern Transformers (BERT, GPT). Probabilistic gating of neuron activation."
    },
    "swish": {
        "fn": swish,
        "derivative": swish_derivative,
        "formula": r"\text{Swish}(z) = z \cdot \sigma(z)",
        "deriv_formula": r"\text{Swish}'(z) = \text{Swish}(z) + \sigma(z)(1 - \text{Swish}(z))",
        "range": r"(-0.28, \infty)",
        "description": "Self-gated smooth activation discovered by Google Brain, often outperforming ReLU on deep models."
    },
    "softmax": {
        "fn": softmax,
        "derivative": softmax_derivative,
        "formula": r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}",
        "deriv_formula": r"\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y",
        "range": r"(0, 1), \sum=1",
        "description": "Converts unnormalized logits to categorical probability distributions."
    },
    "linear": {
        "fn": linear,
        "derivative": linear_derivative,
        "formula": r"f(z) = z",
        "deriv_formula": r"f'(z) = 1",
        "range": r"(-\infty, \infty)",
        "description": "Identity activation for linear layers and continuous regression outputs."
    }
}


def get_activation(name):
    """Retrieve activation function, its derivative and metadata."""
    key = name.lower()
    if key not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: '{name}'. Available: {list(ACTIVATION_REGISTRY.keys())}")
    entry = ACTIVATION_REGISTRY[key]
    return entry["fn"], entry["derivative"]
