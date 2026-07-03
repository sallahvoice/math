# Gaussian Mixture Models

A compact graduate-level educational project that connects Gaussian geometry, latent-variable mixture models, posterior responsibilities, Expectation-Maximization, convergence behavior, and probabilistic classification. The detailed derivations live in `notes/`; notebooks keep formulas brief and focus on intuition, experiments, visual evidence, and implementation.

## Repository structure

```text
gmm/
├── README.md
├── notes/          # mathematical derivations
├── notebooks/      # sequential executable chapters
├── src/            # reusable implementation without plotting
├── figures/        # notebook-generated figures
├── report/         # concise project report
└── requirements.txt
```

## Installation

```bash
cd machine_learning/gmm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook roadmap

1. `01_gaussian_geometry.ipynb` — univariate and multivariate Gaussian geometry, covariance ellipses, eigenvectors, and why one Gaussian can be too restrictive.
2. `02_gaussian_mixture_generation.ipynb` — latent component draws, mixture weights, and synthetic data generation.
3. `03_responsibilities.ipynb` — Bayes-rule posterior probabilities, soft assignment, hard assignment, and confidence.
4. `04_em_algorithm_from_scratch.ipynb` — vectorized EM without `sklearn.mixture.GaussianMixture`.
5. `05_convergence.ipynb` — likelihood curves, local optima, poor initialization, and covariance regularization.
6. `06_classification.ipynb` — GMM decision boundaries and comparison with KMeans and Gaussian Naive Bayes.

Run notebooks in order from the repository root or from `machine_learning/gmm`; each notebook creates `../figures` automatically and saves publication-quality PNG files.

## Implementation details

The `src/` package separates reusable numerical code from notebook storytelling. `gaussian.py` provides Gaussian densities and sampling, `em.py` provides E-step/M-step/log-likelihood utilities, `gmm.py` exposes a small estimator-style class, and `utils.py` contains initialization, regularization, random-seed, and ellipse helpers. The code prioritizes clarity and vectorization where it improves readability.

## Future improvements

- Add diagonal/tied covariance options.
- Add model-selection notebooks for AIC and BIC.
- Add automated notebook execution in CI.
- Extend the report with quantitative benchmark tables from repeated experiments.