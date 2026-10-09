"""
Diagnostic Comparison Script:
Demonstrates Vanishing Gradients (Deep Sigmoid with Standard Normal init)
vs. Healthy Gradients (Deep Swish/ReLU with He init).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from core.network import NeuralNetwork
from core.datasets import generate_dataset
from core.analyser import NetworkHealthAnalyser


def test_vanishing_scenario():
    print("\n" + "=" * 60)
    print("⚠️  SCENARIO 1: Deep Sigmoid with Standard Normal Init (Vanishing Gradients)")
    print("=" * 60)
    X, y, _ = generate_dataset("moons", n_samples=250, noise=0.1, seed=42)

    # 6-layer deep sigmoid network
    net = NeuralNetwork(
        layer_sizes=[2, 16, 16, 16, 16, 1],
        activations=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"],
        init_mode="normal",  # suboptimal init
        loss="bce",
        optimizer="sgd",
        lr=0.1,
        seed=42
    )

    history = net.train(X, y, epochs=50, verbose=False)
    analyser = NetworkHealthAnalyser(net)
    report = analyser.generate_full_report(history)

    print(f"Status: {report['gradient_flow']['status']}")
    print(f"Explanation: {report['gradient_flow']['explanation']}")
    print(f"Layer Grad Norms: {report['gradient_flow']['grad_norms']}")
    print("Recommendations:")
    for r in report["recommendations"]:
        print(f"  - {r}")


def test_healthy_scenario():
    print("\n" + "=" * 60)
    print("✅  SCENARIO 2: Deep GELU with He Normal Init & AdamW (Healthy Gradient Flow)")
    print("=" * 60)
    X, y, _ = generate_dataset("moons", n_samples=250, noise=0.1, seed=42)

    # 6-layer deep GELU network
    net = NeuralNetwork(
        layer_sizes=[2, 16, 16, 16, 16, 1],
        activations=["gelu", "gelu", "gelu", "gelu", "sigmoid"],
        init_mode="he_normal",
        loss="bce",
        optimizer="adamw",
        lr=0.01,
        seed=42
    )

    history = net.train(X, y, epochs=50, verbose=False)
    analyser = NetworkHealthAnalyser(net)
    report = analyser.generate_full_report(history)

    print(f"Status: {report['gradient_flow']['status']}")
    print(f"Explanation: {report['gradient_flow']['explanation']}")
    print(f"Layer Grad Norms: {report['gradient_flow']['grad_norms']}")
    print("Recommendations:")
    for r in report["recommendations"]:
        print(f"  - {r}")


if __name__ == "__main__":
    test_vanishing_scenario()
    test_healthy_scenario()
