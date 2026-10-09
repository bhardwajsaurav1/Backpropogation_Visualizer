"""
First-principles implementations of popular gradient-based optimizers.
No autograd / PyTorch dependencies — state variables (velocities, moments, step counters)
are tracked explicitly with full transparency.
"""

import numpy as np


class Optimizer:
    """Base class for optimizers."""
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.initial_lr = lr
        self.weight_decay = weight_decay
        self.t = 0  # time step / iteration counter

    def update(self, param, grad, state_key):
        """
        Updates param in-place or returns updated param.
        state_key uniquely identifies the parameter (e.g. 'W1', 'b1').
        """
        raise NotImplementedError

    def step_lr(self, epoch, total_epochs, schedule="constant"):
        """Learning rate schedule updater."""
        if schedule == "constant":
            return self.lr
        elif schedule == "step":
            # Halve learning rate every 1/3 of training
            decay_factor = 0.5 ** (epoch // max(1, total_epochs // 3))
            self.lr = self.initial_lr * decay_factor
        elif schedule == "exp":
            self.lr = self.initial_lr * (0.95 ** epoch)
        elif schedule == "cosine":
            self.lr = self.initial_lr * 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))
        return self.lr


class SGD(Optimizer):
    """Vanilla Stochastic / Mini-batch / Full-batch Gradient Descent."""
    def update(self, param, grad, state_key):
        self.t += 1
        # L2 weight decay if enabled
        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param
        delta = -self.lr * grad
        param += delta
        return param, delta


class Momentum(Optimizer):
    """SGD with Polyak Momentum: v = beta * v + lr * grad; param = param - v"""
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta = beta
        self.velocity = {}

    def update(self, param, grad, state_key):
        self.t += 1
        if state_key not in self.velocity:
            self.velocity[state_key] = np.zeros_like(param)

        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param

        v = self.velocity[state_key]
        self.velocity[state_key] = self.beta * v + self.lr * grad
        delta = -self.velocity[state_key]
        param += delta
        return param, delta


class NAG(Optimizer):
    """Nesterov Accelerated Gradient."""
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta = beta
        self.velocity = {}

    def update(self, param, grad, state_key):
        self.t += 1
        if state_key not in self.velocity:
            self.velocity[state_key] = np.zeros_like(param)

        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param

        v_prev = self.velocity[state_key].copy()
        v_next = self.beta * v_prev + self.lr * grad
        self.velocity[state_key] = v_next

        # Nesterov lookahead update
        delta = -(1.0 + self.beta) * v_next + self.beta * v_prev
        param += delta
        return param, delta


class RMSprop(Optimizer):
    """
    RMSprop: Divides gradient by root of exponential moving average of squared gradients.
    s = beta * s + (1 - beta) * grad^2
    param = param - (lr / sqrt(s + eps)) * grad
    """
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, param, grad, state_key):
        self.t += 1
        if state_key not in self.s:
            self.s[state_key] = np.zeros_like(param)

        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param

        self.s[state_key] = self.beta * self.s[state_key] + (1.0 - self.beta) * (grad ** 2)
        step = (self.lr / (np.sqrt(self.s[state_key]) + self.eps)) * grad
        delta = -step
        param += delta
        return param, delta


class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation with first & second moment bias correction.
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param = param - (lr / (sqrt(v_hat) + eps)) * m_hat
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.step_counts = {}

    def update(self, param, grad, state_key):
        if state_key not in self.m:
            self.m[state_key] = np.zeros_like(param)
            self.v[state_key] = np.zeros_like(param)
            self.step_counts[state_key] = 0

        self.step_counts[state_key] += 1
        t = self.step_counts[state_key]

        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param

        # Update biased 1st and 2nd moments
        self.m[state_key] = self.beta1 * self.m[state_key] + (1.0 - self.beta1) * grad
        self.v[state_key] = self.beta2 * self.v[state_key] + (1.0 - self.beta2) * (grad ** 2)

        # Bias corrections
        m_hat = self.m[state_key] / (1.0 - (self.beta1 ** t))
        v_hat = self.v[state_key] / (1.0 - (self.beta2 ** t))

        step = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        delta = -step
        param += delta
        return param, delta


class AdamW(Optimizer):
    """
    AdamW: Adam with decoupled weight decay.
    Weight decay is applied directly to the parameters rather than through gradient moments.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.step_counts = {}

    def update(self, param, grad, state_key):
        if state_key not in self.m:
            self.m[state_key] = np.zeros_like(param)
            self.v[state_key] = np.zeros_like(param)
            self.step_counts[state_key] = 0

        self.step_counts[state_key] += 1
        t = self.step_counts[state_key]

        # First apply decoupled weight decay
        param -= self.lr * self.weight_decay * param

        # Standard Adam on pure gradient
        self.m[state_key] = self.beta1 * self.m[state_key] + (1.0 - self.beta1) * grad
        self.v[state_key] = self.beta2 * self.v[state_key] + (1.0 - self.beta2) * (grad ** 2)

        m_hat = self.m[state_key] / (1.0 - (self.beta1 ** t))
        v_hat = self.v[state_key] / (1.0 - (self.beta2 ** t))

        step = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        delta = -step
        param += delta
        return param, delta


OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSprop,
    "adam": Adam,
    "adamw": AdamW
}


def get_optimizer(name, lr=0.01, **kwargs):
    """Retrieve optimizer instance by name."""
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: '{name}'. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    return OPTIMIZER_REGISTRY[key](lr=lr, **kwargs)
