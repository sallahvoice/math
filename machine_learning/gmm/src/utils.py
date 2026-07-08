"""Utility helpers shared by the GMM implementation and notebooks."""

from __future__ import annotations

import numpy as np


def check_random_state(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a NumPy Generator from an integer seed, Generator, or None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def regularize_covariance(covariance: np.ndarray, reg_covar: float = 1e-6) -> np.ndarray:
    """Add diagonal regularization to a covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)
    return covariance + reg_covar * np.eye(covariance.shape[-1])


def initialize_parameters(X: np.ndarray, n_components: int, random_state: int | np.random.Generator | None = None, reg_covar: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize mixture weights, means, and full covariance matrices."""
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    if n_components > n_samples:
        raise ValueError("n_components cannot exceed n_samples")
    indices = rng.choice(n_samples, size=n_components, replace=False)
    means = X[indices].copy()
    empirical = np.cov(X, rowvar=False)
    if empirical.ndim == 0:
        empirical = empirical.reshape(1, 1)
    covariances = np.repeat(regularize_covariance(empirical, reg_covar)[None, :, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)
    return weights, means, covariances


def covariance_ellipse(covariance: np.ndarray, mean: np.ndarray | None = None, n_std: float = 2.0, n_points: int = 200) -> np.ndarray:
    """Return points on a covariance ellipse for a two-dimensional covariance."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("covariance_ellipse requires a 2x2 covariance")
    mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    transform = vectors @ np.diag(n_std * np.sqrt(np.maximum(values, 0.0)))
    return circle @ transform.T + mean