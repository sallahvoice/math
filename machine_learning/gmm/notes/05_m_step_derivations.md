# 05 — M-Step Derivations

---

## 1. Setup

At the M-step, we maximize the $Q$ function with respect to $\boldsymbol{\theta} = \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k\}$:

$$
Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

where $r_{nk}$ are the responsibilities from the E-step (treated as **constants** during M-step).

Define:
$$
N_k = \sum_{n=1}^{N} r_{nk} \quad \text{(effective number of points in component } k\text{)}
$$

The terms for different $k$ decouple, so we optimize each component independently — except for the constraint $\sum_k \pi_k = 1$.

---

## 2. Deriving $\boldsymbol{\mu}_k$

Expand $\log \mathcal{N}$, keeping only terms that depend on $\boldsymbol{\mu}_k$:

$$
Q \supset -\frac{1}{2} \sum_{n=1}^{N} r_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)
$$

Take the gradient with respect to $\boldsymbol{\mu}_k$ and set to zero:

$$
\nabla_{\boldsymbol{\mu}_k} Q = \sum_{n=1}^{N} r_{nk} \, \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) = \mathbf{0}
$$

Since $\boldsymbol{\Sigma}_k^{-1}$ is invertible, this gives:

$$
\sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n = \boldsymbol{\mu}_k \sum_{n=1}^{N} r_{nk} = \boldsymbol{\mu}_k \, N_k
$$

$$
\boxed{\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n}
$$

**Interpretation:** responsibility-weighted mean of the data points.

---

## 3. Deriving $\boldsymbol{\Sigma}_k$

Let $\mathbf{S} = \boldsymbol{\Sigma}_k$ for brevity. Terms in $Q$ depending on $\mathbf{S}$:

$$
Q \supset \sum_{n=1}^{N} r_{nk} \left[ -\frac{1}{2} \log |\mathbf{S}| - \frac{1}{2} (\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \mathbf{S}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) \right]
$$

$$
= -\frac{N_k}{2} \log |\mathbf{S}| - \frac{1}{2} \sum_{n=1}^{N} r_{nk} \, \text{tr}\left[\mathbf{S}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \right]
$$

Using the matrix identities:
$$
\frac{\partial}{\partial \mathbf{S}} \log |\mathbf{S}| = \mathbf{S}^{-\top} = \mathbf{S}^{-1}, \qquad \frac{\partial}{\partial \mathbf{S}} \text{tr}[\mathbf{S}^{-1} \mathbf{A}] = -\mathbf{S}^{-\top} \mathbf{A}^\top \mathbf{S}^{-\top}
$$

Setting the gradient to zero:

$$
-\frac{N_k}{2} \mathbf{S}^{-1} + \frac{1}{2} \mathbf{S}^{-1} \left(\sum_{n=1}^{N} r_{nk} \, (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \right) \mathbf{S}^{-1} = \mathbf{0}
$$

Multiply both sides by $\mathbf{S}$ from both sides:

$$
\boxed{\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top}
$$

**Interpretation:** responsibility-weighted sample covariance. Uses the **updated** $\boldsymbol{\mu}_k$ (from above).

---

## 4. Deriving $\pi_k$ (with Lagrange Multipliers)

We maximize $Q$ with respect to $\{\pi_k\}$ subject to:
$$
\sum_{k=1}^{K} \pi_k = 1, \qquad \pi_k \geq 0
$$

Terms in $Q$ involving $\pi_k$:

$$
Q \supset \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \log \pi_k = \sum_{k=1}^{K} N_k \log \pi_k
$$

Form the **Lagrangian** with multiplier $\lambda$ for the equality constraint:

$$
\mathcal{L}(\boldsymbol{\pi}, \lambda) = \sum_{k=1}^{K} N_k \log \pi_k + \lambda \left(\sum_{k=1}^{K} \pi_k - 1\right)
$$

Take derivative with respect to $\pi_k$ and set to zero:

$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \implies \pi_k = -\frac{N_k}{\lambda}
$$

Apply the constraint $\sum_k \pi_k = 1$:

$$
\sum_{k=1}^{K} \left(-\frac{N_k}{\lambda}\right) = 1 \implies -\frac{N}{\lambda} = 1 \implies \lambda = -N
$$

Substituting back:

$$
\boxed{\pi_k = \frac{N_k}{N} = \frac{1}{N}\sum_{n=1}^{N} r_{nk}}
$$

**Interpretation:** fraction of the data (by soft count) assigned to component $k$.

---

## 5. Summary of M-Step Updates

Let $N_k = \sum_{n=1}^{N} r_{nk}$. Then:

| Parameter | Update |
|---|---|
| $\boldsymbol{\mu}_k$ | $\dfrac{1}{N_k} \sum_n r_{nk} \mathbf{x}_n$ |
| $\boldsymbol{\Sigma}_k$ | $\dfrac{1}{N_k} \sum_n r_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top$ |
| $\pi_k$ | $\dfrac{N_k}{N}$ |

These are **closed-form** — no inner optimization loop needed. Each update is a weighted version of the MLE formula for a single Gaussian.

---

## 6. Connection to Weighted MLE

If all responsibilities were hard ($r_{nk} \in \{0,1\}$):
- $N_k$ = exact count of points in cluster $k$
- $\boldsymbol{\mu}_k$ = sample mean of those points
- $\boldsymbol{\Sigma}_k$ = sample covariance of those points

EM with soft responsibilities generalizes this: each point contributes to **all** components, weighted by how much it "belongs" to each.