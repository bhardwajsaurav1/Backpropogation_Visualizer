"""
End-to-end integration and convergence tests on non-linear datasets.
"""

import pytest
import numpy as np
from core.network import NeuralNetwork
from core.datasets import generate_dataset
from core.analyser import NetworkHealthAnalyser


def test_xor_convergence():
    """Verifies MLP can easily solve non-linear XOR."""
    X, y, _ = generate_dataset("xor", n_samples=200, noise=0.05, seed=42)
    net = NeuralNetwork(
        layer_sizes=[2, 8, 4, 1],
        activations=["tanh", "tanh", "sigmoid"],
        loss="bce",
        optimizer="adam",
        lr=0.05,
        seed=42
    )

    history = net.train(X, y, epochs=120, verbose=False)
    final_loss = history["loss"][-1]
    final_acc = history["metric"][-1]

    assert final_loss < 0.25, f"XOR loss {final_loss} not sufficiently low"
    assert final_acc >= 0.90, f"XOR accuracy {final_acc} below 90%"


def test_moons_convergence():
    """Verifies convergence on standard Moons dataset."""
    X, y, _ = generate_dataset("moons", n_samples=250, noise=0.1, seed=42)
    net = NeuralNetwork(
        layer_sizes=[2, 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        loss="bce",
        optimizer="adam",
        lr=0.03,
        seed=42
    )

    history = net.train(X, y, epochs=150, verbose=False)
    final_acc = history["metric"][-1]
    assert final_acc >= 0.92, f"Moons accuracy {final_acc} below 92%"


def test_multiclass_spiral():
    """Verifies multi-class Softmax + CCE on 3-arm spiral dataset."""
    X, y, _ = generate_dataset("spiral_3", n_samples=300, noise=0.08, seed=42)
    net = NeuralNetwork(
        layer_sizes=[2, 24, 16, 3],
        activations=["swish", "swish", "softmax"],
        loss="cce",
        optimizer="adam",
        lr=0.02,
        seed=42
    )

    history = net.train(X, y, epochs=200, verbose=False)
    final_acc = history["metric"][-1]
    assert final_acc >= 0.85, f"3-class spiral accuracy {final_acc} below 85%"


def test_health_analyser_report():
    """Verifies that the NetworkHealthAnalyser generates diagnostic outputs."""
    X, y, _ = generate_dataset("moons", n_samples=100, noise=0.1, seed=42)
    net = NeuralNetwork([2, 8, 1], ["relu", "sigmoid"], loss="bce", lr=0.05, seed=42)
    history = net.train(X, y, epochs=20)

    analyser = NetworkHealthAnalyser(net)
    report = analyser.generate_full_report(history)

    assert "gradient_flow" in report
    assert "dead_neurons" in report
    assert "recommendations" in report
    assert report["gradient_flow"]["status"] in ["HEALTHY", "VANISHING", "EXPLODING", "DEGRADED"]
