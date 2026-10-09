"""
Tests for first-principles optimizer updates.
"""

import pytest
import numpy as np
from core.optimizers import SGD, Momentum, NAG, RMSprop, Adam, AdamW


@pytest.mark.parametrize("opt_cls,kwargs,steps", [
    (SGD, {"lr": 0.1}, 100),
    (Momentum, {"lr": 0.1, "beta": 0.9}, 100),
    (NAG, {"lr": 0.1, "beta": 0.9}, 100),
    (RMSprop, {"lr": 0.05, "beta": 0.9}, 150),
    (Adam, {"lr": 0.08, "beta1": 0.9, "beta2": 0.999}, 150),
    (AdamW, {"lr": 0.08, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01}, 150),
])
def test_optimizer_quadratic_minimization(opt_cls, kwargs, steps):
    """Verifies that each optimizer minimizes a simple convex bowl f(w) = w^2."""
    w = np.array([5.0])
    opt = opt_cls(**kwargs)

    for _ in range(steps):
        grad = 2.0 * w
        w, _ = opt.update(w, grad, "w")

    assert np.abs(w[0]) < 0.25, f"Optimizer {opt_cls.__name__} failed to minimize simple quadratic: final w={w[0]}"
