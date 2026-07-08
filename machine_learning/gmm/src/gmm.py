"""A compact full-covariance Gaussian Mixture Model estimator."""

from __future__ import annotations

import numpy as np

from .em import e_step, has_converged, m_step
from .gaussian import sample_mixture
from .utils import check_random_state, initialize_parameters


class GaussianMixtureModel:
    """Full-covariance Gaussian Mixture Model trained with EM."""

    def __init__(self, n_components: int, tol: float = 1e-4, max_iter: int = 100, reg_covar: float = 1e-6, random_state: int | np.random.Generator | None = None):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """Fit the model with Expectation-Maximization."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        rng = check_random_state(self.random_state)
        weights, means, covariances = initialize_parameters(X, self.n_components, rng, self.reg_covar)
        self.lower_bound_history_ = []
        for iteration in range(1, self.max_iter + 1):
            responsibilities, log_prob_norm = e_step(X, weights, means, covariances, self.reg_covar)
            self.lower_bound_history_.append(float(np.sum(log_prob_norm)))
            weights, means, covariances = m_step(X, responsibilities, self.reg_covar)
            if has_converged(self.lower_bound_history_, self.tol):
                break
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.n_iter_ = iteration
        self.converged_ = iteration < self.max_iter
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before prediction")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior component probabilities for each sample."""
        self._check_is_fitted()
        responsibilities, _ = e_step(np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_, self.reg_covar)
        return responsibilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely component index for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample log likelihoods under the mixture."""
        self._check_is_fitted()
        _, log_prob_norm = e_step(np.asarray(X, dtype=float), self.weights_, self.means_, self.covariances_, self.reg_covar)
        return log_prob_norm

    def sample(self, n_samples: int, random_state: int | np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Draw samples from the fitted mixture."""
        self._check_is_fitted()
        return sample_mixture(self.weights_, self.means_, self.covariances_, n_samples, random_state=random_state)