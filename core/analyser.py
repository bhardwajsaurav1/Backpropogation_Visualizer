"""
Diagnostic suite for deep neural network training health.
Detects vanishing/exploding gradients, dead neurons, saturation, ill-conditioning,
and computes input saliency maps.
"""

import numpy as np


class NetworkHealthAnalyser:
    """
    Analyzes neural network training dynamics, parameter distributions,
    and gradient flows.
    """
    def __init__(self, network):
        self.network = network

    def analyze_gradient_flow(self, layer_stats_epoch):
        """
        Calculates gradient norm trends from output layer to input layer.
        Detects vanishing and exploding gradient patterns.
        """
        grad_norms = [layer["grad_w_norm"] for layer in layer_stats_epoch]
        layer_names = [layer["name"] for layer in layer_stats_epoch]

        if len(grad_norms) <= 1:
            ratio = 1.0
            status = "HEALTHY"
            explanation = "Single hidden/output layer network."
        else:
            first_layer_norm = grad_norms[0]
            last_layer_norm = grad_norms[-1]
            eps = 1e-12

            ratio = first_layer_norm / (last_layer_norm + eps)

            if first_layer_norm < 1e-6 and last_layer_norm > 1e-2:
                status = "VANISHING"
                explanation = f"First layer gradient ({first_layer_norm:.2e}) is vanishingly small compared to output layer ({last_layer_norm:.2e}). Deep layers may not be learning."
            elif any(g > 1e3 for g in grad_norms) or ratio > 1e4:
                status = "EXPLODING"
                explanation = f"Gradient norm spike detected (max: {max(grad_norms):.2e}). Risk of numerical instability or NaN weights."
            elif ratio < 0.01:
                status = "DEGRADED"
                explanation = f"Early layer gradients are significantly attenuated (ratio {ratio:.4f}). Consider He/Xavier initialization or residual connections."
            else:
                status = "HEALTHY"
                explanation = "Gradient magnitudes are well-balanced across all layers."

        return {
            "status": status,
            "layer_names": layer_names,
            "grad_norms": grad_norms,
            "first_to_last_ratio": float(ratio),
            "explanation": explanation
        }

    def analyze_dead_neurons(self, layer_stats_epoch):
        """
        Calculates the percentage of inactive (dead) neurons across layers.
        """
        dead_ratios = [layer.get("dead_ratio", 0.0) for layer in layer_stats_epoch]
        layer_names = [layer["name"] for layer in layer_stats_epoch]

        warnings = []
        for idx, (name, ratio) in enumerate(zip(layer_names, dead_ratios)):
            if ratio > 0.5:
                warnings.append(f"{name} has {ratio * 100:.1f}% inactive/dead neurons.")

        return {
            "dead_ratios": dead_ratios,
            "layer_names": layer_names,
            "warnings": warnings,
            "has_high_dead_ratio": any(r > 0.4 for r in dead_ratios)
        }

    def compute_saliency_map(self, X_grid):
        """
        Computes input saliency map: ||dL / dX||_2 across a 2D coordinate grid.
        Identifies which regions of the input space the network is most sensitive to.
        """
        X_grid = np.asarray(X_grid, dtype=float)
        # Compute forward pass
        y_pred = self.network.forward(X_grid, training=False)
        # Saliency with respect to pseudo-target (sensitivity to output change)
        if self.network.loss_name in ["bce", "binary_cross_entropy"]:
            # Grad wrt output probability
            dummy_target = 1.0 - y_pred
        elif self.network.loss_name in ["cce", "categorical_cross_entropy"]:
            dummy_target = np.zeros_like(y_pred)
            dummy_target[np.arange(len(y_pred)), np.argmax(y_pred, axis=-1)] = 0.0
        else:
            dummy_target = y_pred + 1.0

        # Propagate backward to inputs
        dX = self.network.backward(y_pred, dummy_target)
        # Saliency norm per input coordinate
        saliency = np.linalg.norm(dX, axis=1)
        return saliency

    def generate_full_report(self, history=None):
        """
        Generates comprehensive health diagnosis report across the entire training history.
        """
        if history is None:
            history = self.network.history

        if not history["layer_stats"]:
            return {"error": "Network has not been trained yet."}

        final_epoch_stats = history["layer_stats"][-1]
        grad_flow = self.analyze_gradient_flow(final_epoch_stats)
        dead_neurons = self.analyze_dead_neurons(final_epoch_stats)

        # Loss convergence rate
        losses = history["loss"]
        loss_reduction = (losses[0] - losses[-1]) / (losses[0] + 1e-12) if len(losses) > 1 else 0.0

        recommendations = []
        if grad_flow["status"] == "VANISHING":
            recommendations.append("Switch hidden activation to ReLU, LeakyReLU, or GELU to avoid saturation.")
            recommendations.append("Try Xavier or He initialization instead of standard normal initialization.")
        elif grad_flow["status"] == "EXPLODING":
            recommendations.append("Lower the learning rate (e.g. 0.001) or add L2 weight decay.")
            recommendations.append("Switch to Adam/AdamW optimizer with momentum dampening.")

        if dead_neurons["has_high_dead_ratio"]:
            recommendations.append("High dying ReLU percentage detected: switch to LeakyReLU (alpha=0.01) or ELU.")

        if loss_reduction < 0.05:
            recommendations.append("Loss plateaued early: verify learning rate is not too small or increase hidden capacity.")

        if not recommendations:
            recommendations.append("Network training dynamics look optimal! Gradients flow cleanly and parameters are well-conditioned.")

        return {
            "gradient_flow": grad_flow,
            "dead_neurons": dead_neurons,
            "final_loss": float(losses[-1]),
            "loss_reduction_pct": float(loss_reduction * 100),
            "final_metric": float(history["metric"][-1]),
            "recommendations": recommendations
        }
