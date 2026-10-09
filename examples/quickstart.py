"""
Quickstart demonstration of NeuroFlow pure NumPy neural network engine.
Trains a 4-layer MLP on the challenging Spiral dataset and generates a diagnostic report.
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
from core.landscape import LossLandscapeSampler


def main():
    print("=" * 70)
    print("🧠 NeuroFlow: From-Scratch Backpropagation Engine & Health Analyser")
    print("=" * 70)

    # 1. Generate non-linear Spiral dataset (2 classes)
    print("\n[1] Generating 2-Arm Spiral Dataset...")
    X, y, task = generate_dataset("spiral", n_samples=300, noise=0.08, seed=42)
    print(f"    Dataset: X shape = {X.shape}, y shape = {y.shape}, Task = {task}")

    # 2. Construct Deep Neural Network: 2 -> 24 -> 16 -> 8 -> 1
    print("\n[2] Building Deep Neural Architecture:")
    print("    Input (2) -> Dense(24, Swish) -> Dense(16, Swish) -> Dense(8, Tanh) -> Output(1, Sigmoid)")
    net = NeuralNetwork(
        layer_sizes=[2, 24, 16, 8, 1],
        activations=["swish", "swish", "tanh", "sigmoid"],
        init_mode="he_normal",
        loss="bce",
        optimizer="adamw",
        lr=0.02,
        seed=42
    )

    # 3. Train with Full Telemetry Logging
    print("\n[3] Training Network for 150 Epochs...")
    history = net.train(X, y, epochs=150, verbose=True)

    final_loss = history["loss"][-1]
    final_acc = history["metric"][-1]
    print(f"\n    Training Completed! Final Loss: {final_loss:.4f} | Final Accuracy: {final_acc * 100:.1f}%")

    # 4. Run Deep Health Analyser
    print("\n[4] Running Neural Network Health & Gradient Flow Diagnostics...")
    analyser = NetworkHealthAnalyser(net)
    report = analyser.generate_full_report(history)

    grad_flow = report["gradient_flow"]
    print(f"    Gradient Flow Status: [{grad_flow['status']}]")
    print(f"    Explanation: {grad_flow['explanation']}")
    print(f"    First-to-Last Layer Gradient Ratio: {grad_flow['first_to_last_ratio']:.4f}")
    print(f"    Dead Neurons Detected: {report['dead_neurons']['has_high_dead_ratio']}")
    print("    Recommendations:")
    for rec in report["recommendations"]:
        print(f"      • {rec}")

    # 5. Micro-Inspection of a Single Sample (Hand-derived chain rule trace)
    print("\n[5] Micro-Inspecting Sample #0 (Single-Sample Chain Rule Trace):")
    sample_trace = net.inspect_sample(X[0], y[0])
    print(f"    Input x = {sample_trace['x']}")
    print(f"    Target y = {sample_trace['y']}")
    print(f"    Prediction y_hat = {sample_trace['y_pred']}")
    print(f"    Sample Loss = {sample_trace['loss']:.6f}")
    for l_info in sample_trace["layers"]:
        print(f"    Layer {l_info['name']} ({l_info['activation']}):")
        w_shape = np.array(l_info['W']).shape
        grad_w_norm = np.linalg.norm(l_info['grad_W'])
        print(f"      Weights: {w_shape} | ||dL/dW||_2: {grad_w_norm:.6f}")

    # 6. Sample 2D Loss Landscape Surface
    print("\n[6] Sampling 2D Loss Landscape Subspace...")
    sampler = LossLandscapeSampler(net, resolution=9, span=1.0)
    surface_data = sampler.sample_surface(X, y)
    print(f"    Sampled {len(surface_data['alphas'])}x{len(surface_data['betas'])} loss grid successfully!")

    print("\n" + "=" * 70)
    print("✨ Quickstart Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
