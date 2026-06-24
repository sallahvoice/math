# 01 — Gaussian Distributions and Mixture Models

---

## 1. Multivariate Gaussian

A random variable $\mathbf{x} \in \mathbb{R}^D$ follows a multivariate Gaussian:

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

**Parameters:**
- $\boldsymbol{\mu} \in \mathbb{R}^D$ — mean vector (center of the distribution)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ — covariance matrix (must be symmetric positive definite)

---

## 2. Covariance Matrix Interpretation

The covariance matrix $\boldsymbol{\Sigma}$ controls the **shape** and **orientation** of the distribution.

| Structure | Shape | Constraints |
|---|---|---|
| $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ | Spherical (isotropic) | Same variance in all directions |
| $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_D^2)$ | Axis-aligned ellipse | Independent features |
| $\boldsymbol{\Sigma}$ full | Rotated ellipse | Correlated features |

The **Mahalanobis distance** $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ measures distance in the metric defined by $\boldsymbol{\Sigma}$. Points with equal Mahalanobis distance lie on an ellipsoid — the contour lines of the Gaussian.

The **eigendecomposition** $\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$ reveals:
- eigenvectors $\mathbf{U}$: principal axes (orientation)
- eigenvalues $\boldsymbol{\Lambda}$: variance along each axis (size)

---

## 3. Gaussian Mixture Model (GMM)

A GMM represents a distribution as a **weighted sum of $K$ Gaussians**:

$$
p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

**Parameters** $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$:
- $\pi_k$ — mixing coefficient for component $k$
- $\boldsymbol{\mu}_k$ — mean of component $k$
- $\boldsymbol{\Sigma}_k$ — covariance of component $k$

**Constraints on mixing coefficients:**
$$
\pi_k \geq 0, \qquad \sum_{k=1}^{K} \pi_k = 1
$$

So $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ lives on the $(K-1)$-dimensional probability simplex.

---

## 4. Generative View of GMM

A GMM defines a **generative process**:

1. Sample a component index: $z \sim \text{Categorical}(\boldsymbol{\pi})$, so $P(z = k) = \pi_k$
2. Sample a data point: $\mathbf{x} \mid z = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

The marginal over $\mathbf{x}$ recovers the mixture:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} P(z=k) \, p(\mathbf{x} \mid z=k) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

The variable $z$ is **latent** — it exists in the generative story but is never observed. This is the central difficulty. See `02_latent_variables.md`.

---

## 5. Why Mixtures?

A single Gaussian is unimodal and symmetric — it cannot model multimodal or skewed data. A GMM with enough components can approximate **any** continuous density (universal approximator in density estimation).