"""Gaussian density and sampling utilities for the GMM project."""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _as_2d(X: ArrayLike) -> np.ndarray:
    """Return X as a two-dimensional floating point array."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array")
    return X


def log_multivariate_gaussian_pdf(X: ArrayLike, mean: ArrayLike, covariance: ArrayLike, reg_covar: float = 1e-6) -> np.ndarray:
    """Compute log N(x | mean, covariance) for each row of X.

    The implementation uses a Cholesky factorization for numerical stability and
    adds ``reg_covar`` to the diagonal before factorization.
    """
    X = _as_2d(X)
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1:
        raise ValueError("mean must be one-dimensional")
    n_features = mean.shape[0]
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (n_features, n_features):
        raise ValueError("covariance shape must match mean dimensionality")

    cov = covariance + reg_covar * np.eye(n_features)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, reg_covar)
        cov = (eigvecs * eigvals) @ eigvecs.T
        chol = np.linalg.cholesky(cov)

    centered = (X - mean).T
    solved = np.linalg.solve(chol, centered)
    mahalanobis = np.sum(solved * solved, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)


def multivariate_gaussian_pdf(X: ArrayLike, mean: ArrayLike, covariance: ArrayLike, reg_covar: float = 1e-6) -> np.ndarray:
    """Compute N(x | mean, covariance) for each row of X."""
    return np.exp(log_multivariate_gaussian_pdf(X, mean, covariance, reg_covar=reg_covar))


def log_gaussian_matrix(X: ArrayLike, means: ArrayLike, covariances: ArrayLike, reg_covar: float = 1e-6) -> np.ndarray:
    """Return an (n_samples, n_components) matrix of component log densities."""
    X = _as_2d(X)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    return np.column_stack([
        log_multivariate_gaussian_pdf(X, means[k], covariances[k], reg_covar=reg_covar)
        for k in range(means.shape[0])
    ])


def sample_gaussian(mean: ArrayLike, covariance: ArrayLike, n_samples: int, random_state: int | np.random.Generator | None = None) -> np.ndarray:
    """Draw samples from a multivariate Gaussian distribution."""
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    return rng.multivariate_normal(np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float), size=n_samples)


def sample_mixture(weights: ArrayLike, means: ArrayLike, covariances: ArrayLike, n_samples: int, random_state: int | np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw samples and latent labels from a Gaussian mixture."""
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    labels = rng.choice(len(weights), size=n_samples, p=weights)
    X = np.empty((n_samples, np.asarray(means).shape[1]), dtype=float)
    for k in range(len(weights)):
        mask = labels == k
        if np.any(mask):
            X[mask] = rng.multivariate_normal(means[k], covariances[k], size=mask.sum())
    return X, labels