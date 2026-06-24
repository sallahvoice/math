# 06 — GMM for Classification

---

## 1. From Density Estimation to Classification

A fitted GMM gives us $p(\mathbf{x} \mid \boldsymbol{\theta})$. For classification, we fit **one GMM per class** and use Bayes' theorem.

Given classes $c \in \{1, \ldots, C\}$, fit:
$$
p(\mathbf{x} \mid c, \boldsymbol{\theta}_c) = \sum_{k=1}^{K_c} \pi_{ck} \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_{ck}, \boldsymbol{\Sigma}_{ck})
$$

where $K_c$ components are used per class.

---

## 2. Bayes Classification Rule

With class priors $p(c)$ (estimated from training set frequency):

$$
p(c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid c) \, p(c)}{\sum_{c'} p(\mathbf{x} \mid c') \, p(c')}
$$

**Predict** the class with highest posterior:

$$
\hat{c} = \arg\max_c \; p(c \mid \mathbf{x}) = \arg\max_c \; p(\mathbf{x} \mid c) \, p(c)
$$

(The denominator is constant across classes and can be ignored for prediction.)

---

## 3. Soft vs Hard Assignment

**Soft assignment** (default in EM): each point gets a probability of belonging to each component.

$$
r_{nk} = p(z_k = 1 \mid \mathbf{x}_n, \boldsymbol{\theta}) \in [0, 1]
$$

**Hard assignment**: assign each point to its most likely component:

$$
k^* = \arg\max_k \; r_{nk}
$$

This is equivalent to **MAP estimation** of the latent variable, and recovers something like K-means.

| | Soft | Hard |
|---|---|---|
| Assignment | Probabilistic | Deterministic |
| Boundary | Smooth | Sharp |
| Robustness | Better on ambiguous points | Sensitive to boundary points |
| Equivalent to | EM / GMM | K-means (spherical $\boldsymbol{\Sigma}$) |

K-means is a special case of GMM with $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ and $\sigma^2 \to 0$.

---

## 4. Decision Boundaries

The decision boundary between classes $c$ and $c'$ is the set of $\mathbf{x}$ where:

$$
p(c \mid \mathbf{x}) = p(c' \mid \mathbf{x}) \implies p(\mathbf{x} \mid c) \, p(c) = p(\mathbf{x} \mid c') \, p(c')
$$

For GMMs, this boundary is in general **nonlinear** — a mixture of Gaussians can model arbitrarily complex class-conditional distributions, and their intersection can take complex shapes.

Special case: if each class is modeled by a **single Gaussian** with the **same covariance** $\boldsymbol{\Sigma}$, the boundary is a **hyperplane** (linear discriminant analysis).

---

## 5. Generative vs Discriminative

GMM classification is **generative**:
- Models $p(\mathbf{x} \mid c)$ and $p(c)$ separately
- Can generate new samples
- Works with few data per class (can use prior/regularization)
- Less efficient asymptotically than discriminative models

**Discriminative** models (logistic regression, SVM, neural networks):
- Model $p(c \mid \mathbf{x})$ directly
- Cannot generate data
- Typically better classification performance with lots of data

| | Generative (GMM) | Discriminative |
|---|---|---|
| Models | $p(\mathbf{x} \mid c)$, $p(c)$ | $p(c \mid \mathbf{x})$ |
| Data efficiency | Better with few samples | Better with many samples |
| Handles missing data | Yes | Generally no |
| Density estimation | Yes | No |

---

## 6. Anomaly Detection with GMM

A fitted GMM also enables **anomaly detection**: a point $\mathbf{x}$ is anomalous if:

$$
p(\mathbf{x} \mid \boldsymbol{\theta}) < \tau
$$

for some threshold $\tau$. Points in low-density regions of the learned mixture are flagged as outliers. This is a natural extension of the density estimation use case.