"""
Loss Landscape Sampler & Visualizer.
Projects high-dimensional neural network loss functions onto 2D/3D subspace surfaces
using filter-normalized random directions (Li et al., 2018).
"""

import numpy as np


class LossLandscapeSampler:
    """
    Samples loss values in a 2D slice around the network's current or trained parameters.
    """
    def __init__(self, network, resolution=21, span=1.0):
        self.network = network
        self.resolution = resolution
        self.span = span
        self.dir1 = None
        self.dir2 = None
        self.center_weights = None

    def _get_flat_weights(self):
        """Packs all network weights and biases into a 1D vector."""
        params = []
        for dense in self.network.dense_layers:
            params.append(dense.W.ravel())
            params.append(dense.b.ravel())
        return np.concatenate(params)

    def _set_flat_weights(self, flat_params):
        """Unpacks flat 1D vector into network weights and biases."""
        idx = 0
        for dense in self.network.dense_layers:
            w_size = dense.W.size
            b_size = dense.b.size
            dense.W = flat_params[idx : idx + w_size].reshape(dense.W.shape).copy()
            idx += w_size
            dense.b = flat_params[idx : idx + b_size].reshape(dense.b.shape).copy()
            idx += b_size

    def _generate_filter_normalized_direction(self, rng=None):
        """Generates random direction vector with layer-wise filter normalization."""
        if rng is None:
            rng = np.random.default_rng()

        direction = []
        for dense in self.network.dense_layers:
            # Direction for W with matching column norms
            d_W = rng.standard_normal(dense.W.shape)
            w_col_norms = np.linalg.norm(dense.W, axis=0, keepdims=True) + 1e-12
            d_W_col_norms = np.linalg.norm(d_W, axis=0, keepdims=True) + 1e-12
            d_W = d_W * (w_col_norms / d_W_col_norms)
            direction.append(d_W.ravel())

            # Direction for b
            d_b = rng.standard_normal(dense.b.shape)
            b_norm = np.linalg.norm(dense.b) + 1e-12
            d_b_norm = np.linalg.norm(d_b) + 1e-12
            d_b = d_b * (b_norm / d_b_norm)
            direction.append(d_b.ravel())

        return np.concatenate(direction)

    def sample_surface(self, X, y, rng=None):
        """
        Samples the loss on a 2D grid: θ(α, β) = θ_center + α * d1 + β * d2.
        Returns: alphas, betas, loss_surface (2D numpy array).
        """
        original_weights = self._get_flat_weights()
        self.center_weights = original_weights.copy()

        if self.dir1 is None or self.dir2 is None:
            d1 = self._generate_filter_normalized_direction(rng)
            d2 = self._generate_filter_normalized_direction(rng)
            # Gram-Schmidt orthogonalization
            d1 = d1 / (np.linalg.norm(d1) + 1e-12)
            d2 = d2 - np.dot(d1, d2) * d1
            d2 = d2 / (np.linalg.norm(d2) + 1e-12)
            self.dir1 = d1
            self.dir2 = d2

        alphas = np.linspace(-self.span, self.span, self.resolution)
        betas = np.linspace(-self.span, self.span, self.resolution)
        grid_loss = np.zeros((self.resolution, self.resolution))

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = self.center_weights + a * self.dir1 + b * self.dir2
                self._set_flat_weights(perturbed)
                y_pred = self.network.forward(X, training=False)
                loss_val = self.network.compute_loss(y_pred, y)
                grid_loss[j, i] = min(loss_val, 10.0)  # Cap for visualization

        # Restore original weights
        self._set_flat_weights(original_weights)

        return {
            "alphas": alphas.tolist(),
            "betas": betas.tolist(),
            "surface": grid_loss.tolist()
        }

    def project_trajectory(self, weight_history, X, y):
        """
        Projects full training weight trajectory onto the 2D plane (alphas, betas, loss).
        """
        if self.dir1 is None or self.dir2 is None or self.center_weights is None:
            return []

        coords = []
        original = self._get_flat_weights()

        for layer_stats in weight_history:
            flat = []
            for layer in layer_stats:
                flat.append(layer["W"].ravel())
                flat.append(layer["b"].ravel())
            w_t = np.concatenate(flat)

            diff = w_t - self.center_weights
            alpha_t = float(np.dot(diff, self.dir1))
            beta_t = float(np.dot(diff, self.dir2))

            self._set_flat_weights(w_t)
            y_pred = self.network.forward(X, training=False)
            loss_t = float(self.network.compute_loss(y_pred, y))

            coords.append({
                "alpha": alpha_t,
                "beta": beta_t,
                "loss": loss_t
            })

        self._set_flat_weights(original)
        return coords
