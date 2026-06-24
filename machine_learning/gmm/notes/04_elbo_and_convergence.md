# 04 — ELBO and Convergence

---

## 1. Setup

We want to understand **why EM works** — why does maximizing $Q$ at each step guarantee $\ell(\boldsymbol{\theta})$ never decreases?

The answer comes from a lower bound on the log-likelihood called the **ELBO**.

Let $q(\mathbf{Z})$ be any distribution over the latent variables. Then for any $q$:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \text{KL}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))
$$

This decomposition is **exact**. We prove it below.

---

## 2. Jensen's Inequality

For a **concave** function $f$ and a random variable $X$:

$$
f\bigl(\mathbb{E}[X]\bigr) \geq \mathbb{E}[f(X)]
$$

Since $\log$ is concave, Jensen gives:

$$
\log \sum_k \lambda_k x_k \geq \sum_k \lambda_k \log x_k \qquad \text{when } \lambda_k \geq 0, \sum_k \lambda_k = 1
$$

More precisely, for any distribution $q(\mathbf{Z})$:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \log \sum_{\mathbf{Z}} q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \geq \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
$$

The right-hand side is the **ELBO** $\mathcal{L}(q, \boldsymbol{\theta})$.

---

## 3. ELBO Definition

$$
\boxed{\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} = \mathbb{E}_{q}\left[\log \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}\right]}
$$

**ELBO** = Evidence Lower BOund. It lower-bounds the log-evidence (log marginal likelihood):

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) \geq \mathcal{L}(q, \boldsymbol{\theta})
$$

---

## 4. Exact Decomposition

**Claim:**
$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \text{KL}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))
$$

**Proof:**

Start from the ELBO and expand:

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
$$

Use $p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \, p(\mathbf{X} \mid \boldsymbol{\theta})$:

$$
= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \, p(\mathbf{X} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
$$

$$
= \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}{q(\mathbf{Z})} + \sum_{\mathbf{Z}} q(\mathbf{Z}) \log p(\mathbf{X} \mid \boldsymbol{\theta})
$$

$$
= -\text{KL}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})) + \log p(\mathbf{X} \mid \boldsymbol{\theta})
$$

(Since $\sum_{\mathbf{Z}} q(\mathbf{Z}) = 1$ and $\log p(\mathbf{X} \mid \boldsymbol{\theta})$ does not depend on $\mathbf{Z}$.)

Rearranging:

$$
\boxed{\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \text{KL}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))}
$$

Since $\text{KL} \geq 0$ with equality iff $q = p$, we have $\mathcal{L}(q, \boldsymbol{\theta}) \leq \log p(\mathbf{X} \mid \boldsymbol{\theta})$. $\blacksquare$

---

## 5. EM as Coordinate Ascent on the ELBO

EM maximizes $\mathcal{L}(q, \boldsymbol{\theta})$ by alternating over $q$ and $\boldsymbol{\theta}$:

### E-step: optimize over $q$ (holding $\boldsymbol{\theta}^{(t)}$ fixed)

$$
q^* = \arg\max_{q} \; \mathcal{L}(q, \boldsymbol{\theta}^{(t)})
$$

From the decomposition: $\mathcal{L} = \log p(\mathbf{X} \mid \boldsymbol{\theta}) - \text{KL}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))$.

Since $\log p(\mathbf{X} \mid \boldsymbol{\theta}^{(t)})$ is fixed, maximizing $\mathcal{L}$ over $q$ means **minimizing KL**, which is achieved by:

$$
\boxed{q^*(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})}
$$

At this point $\text{KL} = 0$, so the bound is **tight**: $\mathcal{L}(q^*, \boldsymbol{\theta}^{(t)}) = \log p(\mathbf{X} \mid \boldsymbol{\theta}^{(t)})$.

For GMM, $q^*(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})$ factorizes over data points, with $q^*(z_{nk}) = r_{nk}^{(t)}$.

### M-step: optimize over $\boldsymbol{\theta}$ (holding $q^*$ fixed)

$$
\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \; \mathcal{L}(q^*, \boldsymbol{\theta})
$$

With $q^*$ fixed, maximizing $\mathcal{L}(q^*, \boldsymbol{\theta})$ over $\boldsymbol{\theta}$ is equivalent to maximizing $Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$:

$$
\mathcal{L}(q^*, \boldsymbol{\theta}) = \underbrace{\sum_{\mathbf{Z}} q^*(\mathbf{Z}) \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}_{Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})} - \underbrace{\sum_{\mathbf{Z}} q^*(\mathbf{Z}) \log q^*(\mathbf{Z})}_{\text{const w.r.t. } \boldsymbol{\theta}}
$$

---

## 6. Monotonic Improvement Proof

**Theorem:** $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$

**Proof:**

After the E-step at iteration $t$: the bound is tight at $\boldsymbol{\theta}^{(t)}$:

$$
\ell(\boldsymbol{\theta}^{(t)}) = \mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t)}) \qquad \text{(KL = 0)}
$$

After the M-step: $\boldsymbol{\theta}^{(t+1)}$ maximizes $\mathcal{L}(q^{(t)}, \boldsymbol{\theta})$, so:

$$
\mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t)})
$$

Since ELBO is always a lower bound on $\ell$:

$$
\ell(\boldsymbol{\theta}^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \boldsymbol{\theta}^{(t)}) = \ell(\boldsymbol{\theta}^{(t)})
$$

Therefore:

$$
\boxed{\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})}
$$

$\blacksquare$

The log-likelihood is **monotonically non-decreasing** across EM iterations.

---

## 7. Visualizing the ELBO

At each iteration, EM can be visualized as:

1. **E-step:** lift the lower bound so it's tangent to $\ell$ at $\boldsymbol{\theta}^{(t)}$ (KL = 0)
2. **M-step:** slide along the lower bound to find its maximum $\boldsymbol{\theta}^{(t+1)}$
3. The new $\boldsymbol{\theta}^{(t+1)}$ lies higher on $\ell$ than $\boldsymbol{\theta}^{(t)}$

This "tighten then maximize" picture is the geometric core of EM.

---

## 8. ELBO Expanded Form

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \underbrace{\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})]}_{\text{expected complete-data log-likelihood}} + \underbrace{\mathbb{H}[q]}_{\text{entropy of } q}
$$

where $\mathbb{H}[q] = -\sum_{\mathbf{Z}} q(\mathbf{Z}) \log q(\mathbf{Z})$.

In EM (exact inference), $q$ is fixed in the M-step, so entropy is constant and maximizing $\mathcal{L}$ over $\boldsymbol{\theta}$ is exactly maximizing $Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$.

In **variational inference** (approximate EM), $q$ is restricted to a family and both terms matter.