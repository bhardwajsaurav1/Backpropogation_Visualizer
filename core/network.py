"""
Multi-Layer Perceptron (Neural Network) architecture engine.
Coordinates forward propagation, hand-derived backpropagation, optimization steps,
and detailed training telemetry.
"""

import numpy as np
from .layer import DenseLayer, DropoutLayer
from .losses import get_loss
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Arbitrary-depth Fully Connected Neural Network with manual backpropagation.
    """
    def __init__(self, layer_sizes, activations, init_mode="he_normal",
                 loss="bce", optimizer="adam", lr=0.01,
                 dropout_p=0.0, l1=0.0, l2=0.0, seed=42):
        """
        layer_sizes: list of integers, e.g. [2, 16, 8, 1] or [2, 32, 16, 3]
        activations: list of strings (length = len(layer_sizes) - 1), e.g. ['relu', 'relu', 'sigmoid']
                     or single string to apply to all hidden layers with default output activation.
        """
        self.layer_sizes = list(layer_sizes)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.init_mode = init_mode
        self.loss_name = loss.lower()
        self.loss_fn = get_loss(loss)
        self.optimizer_name = optimizer
        self.optimizer = get_optimizer(optimizer, lr=lr) if isinstance(optimizer, str) else optimizer
        self.lr = lr
        self.dropout_p = dropout_p
        self.l1 = l1
        self.l2 = l2

        # Format activations
        num_layers = len(layer_sizes) - 1
        if isinstance(activations, str):
            # If single activation given, treat as hidden activation, pick appropriate output activation
            hidden_act = activations
            if self.loss_name in ["bce", "binary_cross_entropy"]:
                output_act = "sigmoid"
            elif self.loss_name in ["cce", "categorical_cross_entropy"]:
                output_act = "softmax"
            else:
                output_act = "linear"
            self.activations = [hidden_act] * (num_layers - 1) + [output_act]
        else:
            self.activations = list(activations)

        assert len(self.activations) == num_layers, \
            f"Expected {num_layers} activations for {len(layer_sizes)} layer sizes, got {len(self.activations)}"

        # Build layers
        self.layers = []
        self.dense_layers = []

        for i in range(num_layers):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            act = self.activations[i]

            dense = DenseLayer(
                in_features=in_dim,
                out_features=out_dim,
                activation=act,
                init_mode=init_mode,
                l1=l1,
                l2=l2,
                rng=self.rng,
                name=f"Dense_{i+1}"
            )
            self.layers.append(dense)
            self.dense_layers.append(dense)

            # Insert Dropout after hidden dense layers if specified
            if dropout_p > 0.0 and i < num_layers - 1:
                drop = DropoutLayer(p=dropout_p, rng=self.rng)
                self.layers.append(drop)

        # Full training history and telemetry
        self.history = {
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "metric": [],  # accuracy or R2
            "val_metric": [],
            "layer_stats": [],  # stores stats for each dense layer at each epoch
            "lr": []
        }

    def forward(self, X, training=True):
        """Forward pass through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, y_pred, y_true):
        """
        Backpropagation through all layers using hand-derived matrix chain rule.
        Uses exact analytical gradients for output activations + loss combinations.
        """
        m = y_true.shape[0]

        # Handle simplified output layer gradient dL/dZ directly for standard pairs
        last_dense = self.dense_layers[-1]
        out_act = last_dense.activation_name

        if (out_act == "sigmoid" and self.loss_name in ["bce", "binary_cross_entropy"]):
            # Combined Sigmoid + BCE gradient: dL/dZ = (1/m) * (y_pred - y_true)
            dZ = (1.0 / m) * (y_pred - y_true)
            upstream_grad = dZ
            is_direct_dZ = True
        elif (out_act == "softmax" and self.loss_name in ["cce", "categorical_cross_entropy"]):
            # Combined Softmax + CCE gradient: dL/dZ = (1/m) * (y_pred - y_true)
            if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1 and y_pred.shape[1] > 1):
                y_indices = y_true.ravel().astype(int)
                one_hot = np.zeros_like(y_pred)
                one_hot[np.arange(len(y_indices)), y_indices] = 1.0
                y_true = one_hot
            dZ = (1.0 / m) * (y_pred - y_true)
            upstream_grad = dZ
            is_direct_dZ = True
        elif (out_act == "linear" and self.loss_name in ["mse", "mean_squared_error"]):
            # Combined Linear + MSE gradient: dL/dZ = (1/m) * (y_pred - y_true)
            dZ = (1.0 / m) * (y_pred - y_true)
            upstream_grad = dZ
            is_direct_dZ = True
        else:
            # General chain rule: dL/dA from loss backward
            upstream_grad = self.loss_fn.backward(y_pred, y_true)
            is_direct_dZ = False

        # Propagate backward through all layers in reverse order
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                upstream_grad = layer.backward(upstream_grad, is_direct_dZ=is_direct_dZ)
                is_direct_dZ = False  # Only the last layer might receive direct dZ
            elif isinstance(layer, DropoutLayer):
                upstream_grad = layer.backward(upstream_grad)

        return upstream_grad

    def update_parameters(self):
        """Applies optimizer update step to all trainable parameters."""
        deltas = {}
        for idx, dense in enumerate(self.dense_layers):
            w_key = f"W{idx+1}"
            b_key = f"b{idx+1}"
            dense.W, dW = self.optimizer.update(dense.W, dense.grad_W, w_key)
            dense.b, db = self.optimizer.update(dense.b, dense.grad_b, b_key)
            deltas[w_key] = dW
            deltas[b_key] = db
        return deltas

    def compute_loss(self, y_pred, y_true):
        """Computes data loss + L1/L2 regularization."""
        base_loss = self.loss_fn.forward(y_pred, y_true)
        reg_loss = sum(dense.get_regularization_loss() for dense in self.dense_layers)
        return base_loss + reg_loss

    def compute_metric(self, y_pred, y_true):
        """Computes Accuracy for classification, or R2/Negative MSE for regression."""
        if self.loss_name in ["bce", "binary_cross_entropy"]:
            preds = (y_pred >= 0.5).astype(int)
            return float(np.mean(preds == y_true.astype(int)))
        elif self.loss_name in ["cce", "categorical_cross_entropy"]:
            pred_classes = np.argmax(y_pred, axis=-1)
            true_classes = y_true if y_true.ndim == 1 else np.argmax(y_true, axis=-1)
            return float(np.mean(pred_classes == true_classes))
        else:
            # R2 score for regression
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            if ss_tot < 1e-12:
                return 1.0
            return float(1.0 - (ss_res / ss_tot))

    def train_step(self, X_batch, y_batch):
        """Performs a single forward, backward, and parameter update pass."""
        y_pred = self.forward(X_batch, training=True)
        loss = self.compute_loss(y_pred, y_batch)
        metric = self.compute_metric(y_pred, y_batch)
        self.backward(y_pred, y_batch)
        deltas = self.update_parameters()
        return loss, metric, deltas

    def train(self, X, y, X_val=None, y_val=None, epochs=200, batch_size=None,
              lr_schedule="constant", verbose=False):
        """
        Trains the neural network with telemetry logging.
        batch_size: if None, performs full-batch gradient descent.
        """
        n_samples = X.shape[0]
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        for epoch in range(1, epochs + 1):
            # Update learning rate schedule
            current_lr = self.optimizer.step_lr(epoch, epochs, schedule=lr_schedule)

            # Shuffle dataset for mini-batch training
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            epoch_metrics = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_b = X_shuffled[start_idx:end_idx]
                y_b = y_shuffled[start_idx:end_idx]

                b_loss, b_metric, _ = self.train_step(X_b, y_b)
                epoch_losses.append(b_loss)
                epoch_metrics.append(b_metric)

            # Full evaluation on train and validation sets at end of epoch
            train_pred = self.forward(X, training=False)
            train_loss = self.compute_loss(train_pred, y)
            train_metric = self.compute_metric(train_pred, y)

            val_loss = None
            val_metric = None
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(val_pred, y_val)
                val_metric = self.compute_metric(val_pred, y_val)

            # Collect telemetry per dense layer
            layer_info = []
            for idx, dense in enumerate(self.dense_layers):
                w_norm = float(np.linalg.norm(dense.W))
                b_norm = float(np.linalg.norm(dense.b))
                grad_w_norm = float(np.linalg.norm(dense.grad_W))
                grad_b_norm = float(np.linalg.norm(dense.grad_b))

                # Dead neuron ratio for ReLU / LeakyReLU activations
                act_cache = dense.cache.get("A", None)
                dead_ratio = 0.0
                if act_cache is not None:
                    if dense.activation_name == "relu":
                        dead_ratio = float(np.mean(np.all(act_cache <= 0.0, axis=0)))
                    elif dense.activation_name == "sigmoid":
                        # Saturation: |a - 0.5| > 0.45
                        dead_ratio = float(np.mean(np.abs(act_cache - 0.5) > 0.45))
                    elif dense.activation_name == "tanh":
                        # Saturation: |a| > 0.95
                        dead_ratio = float(np.mean(np.abs(act_cache) > 0.95))

                layer_info.append({
                    "name": dense.name,
                    "in_dim": dense.in_features,
                    "out_dim": dense.out_features,
                    "activation": dense.activation_name,
                    "W": dense.W.copy(),
                    "b": dense.b.copy(),
                    "grad_W": dense.grad_W.copy(),
                    "grad_b": dense.grad_b.copy(),
                    "w_norm": w_norm,
                    "b_norm": b_norm,
                    "grad_w_norm": grad_w_norm,
                    "grad_b_norm": grad_b_norm,
                    "dead_ratio": dead_ratio,
                    "mean_act": float(np.mean(act_cache)) if act_cache is not None else 0.0,
                    "std_act": float(np.std(act_cache)) if act_cache is not None else 0.0
                })

            self.history["epoch"].append(epoch)
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["metric"].append(train_metric)
            self.history["val_metric"].append(val_metric)
            self.history["layer_stats"].append(layer_info)
            self.history["lr"].append(current_lr)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {train_loss:.4f} | Metric: {train_metric:.4f}")

        return self.history

    def inspect_sample(self, x_single, y_single):
        """
        Executes a single-sample forward and backward pass, returning
        complete intermediate numerical steps for micro-inspections and chain-rule breakdowns.
        """
        if x_single.ndim == 1:
            x_single = x_single.reshape(1, -1)
        if y_single.ndim == 1 and self.loss_name in ["bce", "binary_cross_entropy", "mse"]:
            y_single = y_single.reshape(1, -1)

        y_pred = self.forward(x_single, training=False)
        loss_val = self.loss_fn.forward(y_pred, y_single)
        self.backward(y_pred, y_single)

        trace = {
            "x": x_single.tolist(),
            "y": y_single.tolist(),
            "y_pred": y_pred.tolist(),
            "loss": loss_val,
            "layers": []
        }

        for idx, dense in enumerate(self.dense_layers):
            c = dense.cache
            trace["layers"].append({
                "layer_index": idx,
                "name": dense.name,
                "activation": dense.activation_name,
                "X_in": c["X"].tolist(),
                "W": dense.W.tolist(),
                "b": dense.b.tolist(),
                "Z": c["Z"].tolist(),
                "A": c["A"].tolist(),
                "dZ": c.get("dZ", np.zeros_like(c["Z"])).tolist(),
                "grad_W": dense.grad_W.tolist(),
                "grad_b": dense.grad_b.tolist(),
                "dX": c.get("dX", np.zeros_like(c["X"])).tolist()
            })

        return trace

    def predict(self, X):
        """Returns predicted class labels or continuous values."""
        y_pred = self.forward(X, training=False)
        if self.loss_name in ["bce", "binary_cross_entropy"]:
            return (y_pred >= 0.5).astype(int)
        elif self.loss_name in ["cce", "categorical_cross_entropy"]:
            return np.argmax(y_pred, axis=-1)
        else:
            return y_pred

    def predict_proba(self, X):
        """Returns predicted probability scores."""
        return self.forward(X, training=False)
