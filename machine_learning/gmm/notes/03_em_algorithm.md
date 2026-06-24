# 03 — The EM Algorithm

---

## 1. The Problem

We want to maximize the **observed-data log-likelihood**:

$$
\ell(\boldsymbol{\theta}) = \log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Direct maximization is intractable because of the $\log \sum$. EM sidesteps this by working with the **complete-data likelihood** instead.

---

## 2. Complete-Data Likelihood

If we **knew** all assignments $\mathbf{Z} = \{z_{nk}\}$, the complete-data log-likelihood would be:

$$
\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

This is a **sum of logs** (not a log of sums), which is easy to maximize. EM's idea: replace unknown $z_{nk}$ with their expected values under the current parameters.

---

## 3. The EM Framework

**Goal:** maximize $\ell(\boldsymbol{\theta})$ via a sequence $\boldsymbol{\theta}^{(0)}, \boldsymbol{\theta}^{(1)}, \ldots$ such that $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$.

EM alternates between two steps:

### E-Step (Expectation)

Compute the **expected complete-data log-likelihood** under the posterior $p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})$:

$$
Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}} \left[ \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \right]
$$

For GMM, this reduces to computing **responsibilities**:

$$
\boxed{r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}}
$$

Because $\mathbb{E}[z_{nk}] = r_{nk}$, we simply substitute $z_{nk} \leftarrow r_{nk}$ in the complete-data log-likelihood:

$$
Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk}^{(t)} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

### M-Step (Maximization)

Update parameters by maximizing $Q$:

$$
\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \; Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})
$$

Because $Q$ decomposes over $k$, each component can be optimized independently.

**Define effective count for component $k$:**
$$
N_k = \sum_{n=1}^{N} r_{nk}
$$

The M-step updates are (derivations in `05_m_step_derivations.md`):

$$
\boxed{\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, \mathbf{x}_n}
$$

$$
\boxed{\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} \, (\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top}
$$

$$
\boxed{\pi_k^{(t+1)} = \frac{N_k}{N}}
$$

---

## 4. The EM Loop (Algorithm)

```
Initialize θ⁽⁰⁾ = {π_k, μ_k, Σ_k} for k = 1…K

Repeat until convergence:
    E-step: compute r_{nk} for all n, k  using θ⁽ᵗ⁾
    M-step: update μ_k, Σ_k, π_k        using r_{nk}
    Evaluate ℓ(θ⁽ᵗ⁺¹⁾)

Convergence: |ℓ(θ⁽ᵗ⁺¹⁾) - ℓ(θ⁽ᵗ⁾)| < ε
```

---

## 5. Responsibilities: Intuition

$r_{nk}$ is the **soft assignment** of point $\mathbf{x}_n$ to component $k$.

- If $r_{nk} \approx 1$: point $n$ almost certainly belongs to component $k$
- If $r_{nk} \approx 1/K$: point $n$ is ambiguous between components

The M-step updates are **responsibility-weighted statistics** — a soft version of computing means and covariances per cluster.

---

## 6. Initialization

EM is sensitive to initialization because $\ell(\boldsymbol{\theta})$ is non-convex with multiple local maxima.

Common strategies:
- **Random initialization:** assign $\boldsymbol{\mu}_k$ to random data points
- **K-means warm start:** use K-means cluster centers as initial $\boldsymbol{\mu}_k$
- **Multiple restarts:** run EM $R$ times, keep the solution with highest $\ell$

---

## 7. Convergence

EM is **guaranteed to not decrease** $\ell(\boldsymbol{\theta})$ at each iteration (proof in `04_elbo_and_convergence.md`). It converges to a **local maximum** (or saddle point) of the observed-data log-likelihood.

It does **not** guarantee finding the global maximum.