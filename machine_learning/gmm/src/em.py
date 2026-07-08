"""Expectation-Maximization primitives for full-covariance GMMs."""

from __future__ import annotations

import numpy as np

from .gaussian import log_gaussian_matrix
from .utils import regularize_covariance


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Small NumPy implementation of scipy.special.logsumexp."""
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, reg_covar: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Compute responsibilities and per-sample log probabilities."""
    log_prob = log_gaussian_matrix(X, means, covariances, reg_covar=reg_covar)
    weighted_log_prob = log_prob + np.log(np.asarray(weights, dtype=float) + 1e-300)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    responsibilities = np.exp(weighted_log_prob - log_prob_norm[:, None])
    return responsibilities, log_prob_norm


def m_step(X: np.ndarray, responsibilities: np.ndarray, reg_covar: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update weights, means, and covariances from responsibilities."""
    X = np.asarray(X, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    n_samples, n_features = X.shape
    nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
    weights = nk / n_samples
    means = (responsibilities.T @ X) / nk[:, None]
    covariances = np.empty((responsibilities.shape[1], n_features, n_features), dtype=float)
    for k in range(responsibilities.shape[1]):
        diff = X - means[k]
        covariances[k] = (responsibilities[:, k][:, None] * diff).T @ diff / nk[k]
        covariances[k] = regularize_covariance(covariances[k], reg_covar)
    return weights, means, covariances


def compute_log_likelihood(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, reg_covar: float = 1e-6) -> float:
    """Return the total observed-data log-likelihood."""
    _, log_prob_norm = e_step(X, weights, means, covariances, reg_covar=reg_covar)
    return float(np.sum(log_prob_norm))


def has_converged(history: list[float], tol: float = 1e-4) -> bool:
    """Return True when successive likelihood improvements are below tol."""
    return len(history) >= 2 and abs(history[-1] - history[-2]) < tol