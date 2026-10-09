"""
NeuroFlow: From-scratch Neural Network & Backpropagation Engine.
Transparent, hand-derived calculus with complete micro-step tracing and health diagnostics.
"""

from .activations import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    elu, elu_derivative,
    gelu, gelu_derivative,
    swish, swish_derivative,
    softmax, softmax_derivative,
    linear, linear_derivative,
    get_activation
)

from .losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
    HuberLoss,
    get_loss
)

from .optimizers import (
    SGD,
    Momentum,
    NAG,
    RMSprop,
    Adam,
    AdamW,
    get_optimizer
)

from .layer import DenseLayer, DropoutLayer
from .network import NeuralNetwork
from .analyser import NetworkHealthAnalyser
from .landscape import LossLandscapeSampler
from .datasets import generate_dataset

__version__ = "1.0.0"
__all__ = [
    "sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "swish", "softmax", "linear",
    "get_activation", "BinaryCrossEntropy", "CategoricalCrossEntropy", "MeanSquaredError", "HuberLoss",
    "get_loss", "SGD", "Momentum", "NAG", "RMSprop", "Adam", "AdamW", "get_optimizer",
    "DenseLayer", "DropoutLayer", "NeuralNetwork", "NetworkHealthAnalyser",
    "LossLandscapeSampler", "generate_dataset"
]
