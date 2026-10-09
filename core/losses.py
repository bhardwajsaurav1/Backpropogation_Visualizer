"""
Loss functions and their analytical gradients.
Provides both loss value and hand-derived backward gradient wrt predicted values and pre-activation logits.
"""

import numpy as np


class Loss:
    """Base Loss class."""
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        """Returns dLoss / dy_pred."""
        raise NotImplementedError


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss for 2-class classification.
    L = - (1/m) * sum( y * log(y_pred) + (1-y) * log(1 - y_pred) )
    """
    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_pred, y_true):
        """
        dL / dy_pred = (1/m) * ( (y_pred - y) / (y_pred * (1 - y_pred)) )
        """
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        m = y_true.shape[0]
        return (1.0 / m) * (y_pred - y_true) / (y_pred * (1.0 - y_pred))


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss for multi-class classification.
    L = - (1/m) * sum_i sum_k y_{i,k} * log(y_pred_{i,k})
    """
    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        # If y_true is 1D class indices, convert to one-hot
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1 and y_pred.shape[1] > 1):
            y_indices = y_true.ravel().astype(int)
            num_classes = y_pred.shape[1]
            one_hot = np.zeros_like(y_pred)
            one_hot[np.arange(len(y_indices)), y_indices] = 1.0
            y_true = one_hot

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return float(loss)

    def backward(self, y_pred, y_true):
        """
        dL / dy_pred = - (1/m) * (y_true / y_pred)
        Note: When combined with Softmax, dL / dz = (1/m) * (y_pred - y_true).
        """
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1 and y_pred.shape[1] > 1):
            y_indices = y_true.ravel().astype(int)
            one_hot = np.zeros_like(y_pred)
            one_hot[np.arange(len(y_indices)), y_indices] = 1.0
            y_true = one_hot

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        m = y_true.shape[0]
        return -(1.0 / m) * (y_true / y_pred)


class MeanSquaredError(Loss):
    """
    Mean Squared Error Loss for regression:
    L = (1 / (2*m)) * sum((y_pred - y_true)^2)
    """
    def forward(self, y_pred, y_true):
        m = y_true.shape[0]
        return float(np.sum((y_pred - y_true) ** 2) / (2.0 * m))

    def backward(self, y_pred, y_true):
        """
        dL / dy_pred = (1/m) * (y_pred - y_true)
        """
        m = y_true.shape[0]
        return (1.0 / m) * (y_pred - y_true)


class HuberLoss(Loss):
    """
    Huber (Smooth L1) Loss:
    L = 0.5 * (y_pred - y)^2 if |y_pred - y| <= delta else delta * (|y_pred - y| - 0.5 * delta)
    Robust against outliers.
    """
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * (quadratic ** 2) + self.delta * linear
        return float(np.mean(loss))

    def backward(self, y_pred, y_true):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        m = y_true.shape[0]
        grad = np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff))
        return (1.0 / m) * grad


LOSS_REGISTRY = {
    "bce": BinaryCrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "huber": HuberLoss
}


def get_loss(name, **kwargs):
    """Retrieve loss instance by name."""
    key = name.lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[key](**kwargs)
