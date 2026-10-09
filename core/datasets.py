"""
Synthetic 2D dataset generators for classification and regression tasks.
Pure NumPy implementations with zero external dependencies required for generation.
"""

import numpy as np


def make_moons(n_samples=250, noise=0.1, seed=0):
    """Two interleaving half circles."""
    rng = np.random.default_rng(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1.0 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1.0 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.column_stack([outer_circ_x, outer_circ_y]),
        np.column_stack([inner_circ_x, inner_circ_y])
    ])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X, y


def make_circles(n_samples=250, noise=0.1, factor=0.5, seed=0):
    """A large circle containing a smaller circle in 2D."""
    rng = np.random.default_rng(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)

    outer_x = np.cos(linspace_out)
    outer_y = np.sin(linspace_out)
    inner_x = np.cos(linspace_in) * factor
    inner_y = np.sin(linspace_in) * factor

    X = np.vstack([
        np.column_stack([outer_x, outer_y]),
        np.column_stack([inner_x, inner_y])
    ])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X, y


def make_spiral(n_samples=300, noise=0.1, n_classes=2, seed=0):
    """Archimedean spirals with arbitrary number of arms (classes)."""
    rng = np.random.default_rng(seed)
    n_points_per_class = n_samples // n_classes

    X = []
    y = []

    for c in range(n_classes):
        r = np.linspace(0.1, 1.0, n_points_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n_points_per_class) + rng.normal(0, noise, n_points_per_class)
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.column_stack([x1, x2]))
        y.append(np.full(n_points_per_class, c))

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def make_xor(n_samples=250, noise=0.1, seed=0):
    """XOR quadrant problem."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, (n_samples, 2))
    y = ((pts[:, 0] > 0) ^ (pts[:, 1] > 0)).astype(float)
    if noise > 0:
        pts += rng.normal(0, noise, size=pts.shape)
    return pts, y


def make_blobs(n_samples=250, centers=2, noise=0.15, seed=0):
    """Isotropic Gaussian blobs for clustering/classification."""
    rng = np.random.default_rng(seed)
    n_per_center = n_samples // centers
    X = []
    y = []

    # Choose fixed center coordinates
    center_coords = [
        [-0.6, -0.6], [0.6, 0.6], [-0.6, 0.6], [0.6, -0.6]
    ]

    for c in range(centers):
        loc = center_coords[c % len(center_coords)]
        pts = rng.normal(loc=loc, scale=noise + 0.1, size=(n_per_center, 2))
        X.append(pts)
        y.append(np.full(n_per_center, c))

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def make_sine_regression(n_samples=200, noise=0.1, seed=0):
    """1D/2D Sine wave regression dataset."""
    rng = np.random.default_rng(seed)
    x1 = np.linspace(-3.0, 3.0, n_samples)
    x2 = rng.uniform(-1.0, 1.0, n_samples)
    y = np.sin(x1) * np.cos(x2 * 0.5)
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    X = np.column_stack([x1, x2])
    return X, y.reshape(-1, 1)


def generate_dataset(name="moons", n_samples=250, noise=0.1, normalize=True, seed=0):
    """
    Standard dataset factory function.
    Returns: X (normalized 2D coords), y (labels or target values).
    """
    key = name.lower()
    if key == "moons":
        X, y = make_moons(n_samples, noise, seed)
        task = "binary"
    elif key == "circles":
        X, y = make_circles(n_samples, noise, seed=seed)
        task = "binary"
    elif key in ["spiral", "spiral_2"]:
        X, y = make_spiral(n_samples, noise, n_classes=2, seed=seed)
        task = "binary"
    elif key in ["spiral_3", "spiral_3class"]:
        X, y = make_spiral(n_samples, noise, n_classes=3, seed=seed)
        task = "multiclass"
    elif key == "xor":
        X, y = make_xor(n_samples, noise, seed)
        task = "binary"
    elif key in ["blobs", "blobs_2"]:
        X, y = make_blobs(n_samples, centers=2, noise=noise, seed=seed)
        task = "binary"
    elif key == "blobs_3":
        X, y = make_blobs(n_samples, centers=3, noise=noise, seed=seed)
        task = "multiclass"
    elif key in ["sine", "regression"]:
        X, y = make_sine_regression(n_samples, noise, seed)
        task = "regression"
    else:
        X, y = make_moons(n_samples, noise, seed)
        task = "binary"

    if normalize and task != "regression":
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-12
        X = (X - mean) / std

    # Reshape binary labels to column vector (N, 1)
    if task == "binary" and y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y, task
