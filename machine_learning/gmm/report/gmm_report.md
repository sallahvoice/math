# Gaussian Mixture Models: Geometry, Inference, and EM

## 1. Introduction

This project presents Gaussian Mixture Models (GMMs) as a connected sequence of geometric, probabilistic, and computational ideas. The goal is not merely to provide a working implementation, but to explain why the implementation has the form it does and how its behavior can be inspected visually. The detailed mathematical derivations are intentionally kept in the `notes/` directory. The notebooks and this report summarize only the formulas needed to understand the experiments.

A single Gaussian distribution is often the first continuous probability model students encounter. It is analytically convenient, geometrically interpretable, and easy to sample from. Its limitation is equally important: one Gaussian describes one elliptical region of high probability. Many real datasets contain multiple groups, skewed aggregate shapes, or class-conditional subpopulations. GMMs address this by combining several local Gaussian models with mixture weights.

The project is organized as a progressive story: Gaussian geometry motivates mixtures; mixtures introduce latent component labels; latent labels lead to responsibilities; responsibilities lead naturally to the Expectation-Maximization algorithm; convergence experiments reveal practical limitations; and the final notebook uses the same posterior probabilities for classification.

## 2. Gaussian distributions

The first notebook studies univariate and multivariate Gaussian distributions as geometric objects. In one dimension, changing the mean translates the density, while changing the standard deviation spreads or concentrates probability mass. The area under the density remains one because the normalizing constant changes with the scale.

In two dimensions, the covariance matrix determines the orientation and eccentricity of the density contours. Identity covariance gives circular contours. Unequal variances along coordinate axes produce axis-aligned ellipses. Nonzero off-diagonal covariance produces correlated directions, so the ellipse rotates. Eigendecomposition makes this visual: eigenvectors point along the principal axes, and eigenvalues determine the squared lengths along those axes.

This geometry is essential for GMMs. Each component contributes one local elliptical explanation of the data. A dataset that cannot be summarized by one ellipse may still be well approximated by several ellipses placed at different locations.

## 3. Gaussian Mixture Models

A Gaussian mixture density is a weighted sum of component Gaussian densities. The weights are nonnegative and sum to one, so they define a categorical distribution over components. A useful generative interpretation has two steps. First, draw an unobserved component label from the categorical distribution. Second, draw the observation from the Gaussian associated with that label.

This latent-variable view explains why a mixture can be multimodal even though every individual component is unimodal. It also clarifies the role of mixture weights: weights determine how frequently components appear in samples, not merely how strongly they are colored in a plot. The second notebook uses synthetic data so that the hidden component labels are known during generation and can be compared with the unlabeled aggregate distribution.

## 4. Latent variables

In real applications the component label is rarely observed. Instead, each observation has posterior probabilities over components. These probabilities are called responsibilities because they measure how responsible each component is for explaining each point.

Responsibilities are computed with Bayes' theorem. The numerator multiplies a prior mixture weight by the likelihood of the point under a component. The denominator sums this quantity over all components so that the responsibilities for a point sum to one. This creates a soft assignment. A hard cluster label can be obtained by taking the largest responsibility, but doing so discards useful uncertainty.

The responsibility visualizations emphasize regions where the model is confident and regions where components overlap. Points near overlap boundaries may have ambiguous memberships, and this ambiguity is precisely what EM uses instead of prematurely committing to hard labels.

## 5. Expectation-Maximization overview

The EM algorithm alternates between estimating latent structure and updating parameters. In the E-step, current parameters produce responsibilities for every observation-component pair. In the M-step, those responsibilities act like fractional counts in weighted estimates of component weights, means, and covariance matrices.

The observed-data log-likelihood provides the main convergence diagnostic. A correctly implemented EM procedure should not decrease this quantity after a full E/M update cycle, although numerical issues and stopping conventions can affect very small changes. The fourth notebook implements the complete procedure from scratch using NumPy and deliberately avoids `sklearn.mixture.GaussianMixture` so that every intermediate object remains inspectable.

The implementation separates plotting from reusable numerical code. Gaussian density evaluation and sampling live in `src/gaussian.py`; E-step, M-step, and likelihood utilities live in `src/em.py`; the estimator-style interface lives in `src/gmm.py`; and initialization, regularization, and ellipse helpers live in `src/utils.py`.

## 6. Experimental setup

The experiments use reproducible two-dimensional synthetic datasets. Two dimensions are not chosen because GMMs are restricted to the plane; they are chosen because covariance geometry, posterior uncertainty, and decision boundaries are directly visible. Every notebook sets a random seed and saves figures into `figures/` with consistent names and transparent backgrounds.

The main experimental themes are:

- how Gaussian parameters alter density geometry;
- how mixture weights affect empirical cluster proportions;
- how responsibilities vary smoothly across space;
- how EM assignments evolve from initialization to convergence;
- how random initialization and component count influence local optima;
- how GMM decision boundaries compare with KMeans and Gaussian Naive Bayes.

## 7. Results

The Gaussian geometry experiments show that the covariance matrix is not a technical detail; it is the object that determines what shape the model can represent. Spherical covariance captures round clusters, diagonal anisotropic covariance captures axis-aligned stretching, and full covariance captures rotated correlation structure.

The mixture generation experiments show that a weighted sum of simple Gaussian components can produce aggregate distributions that are not themselves Gaussian. Ground-truth labels make the latent process visible, while the unlabeled scatter plot shows the problem that inference must solve.

The responsibility experiments show that posterior probabilities provide more information than hard labels. Confidence is high far from overlap regions and lower near boundaries. This interpretation is important for both clustering and classification because it distinguishes uncertain points from confidently assigned points.

The EM experiments show rapid early improvement followed by slower refinement. Visual snapshots at initialization, iteration 5, iteration 10, and the final iteration illustrate how soft assignments and covariance estimates stabilize together. The log-likelihood curve is a compact numerical summary of this process.

The convergence experiments demonstrate that EM is sensitive to initialization and model specification. Different random starts can converge to different local optima. Too few components underfit multimodal structure, while too many components may assign a component to a small subset of points. Covariance regularization is therefore both a numerical safeguard and a modeling discipline.

The classification experiments show that GMMs can be used beyond unsupervised density estimation. Posterior probabilities become probabilistic predictions, and the induced decision boundary can bend according to covariance structure. KMeans produces distance-based partitions, while Gaussian Naive Bayes uses supervised labels but assumes conditional feature independence.

## 8. Discussion

The project prioritizes clarity over maximum performance. Full vectorization is used where it makes the mathematics clearer, such as evaluating component log densities for all samples and normalizing responsibilities. Small loops are retained where they improve readability, especially for covariance updates across components.

The notebooks are intentionally not derivation-heavy. They provide enough formula context to run and interpret each experiment, then point back to the notes for full derivations. This keeps the notebooks focused on the questions that plots can answer: What changed geometrically? Where is the model uncertain? Did likelihood improve? How did initialization affect the result?

## 9. Limitations

The implementation supports full covariance matrices but does not yet provide diagonal, spherical, or tied covariance options. It does not implement model-selection criteria such as AIC or BIC. The experiments focus on clean synthetic data rather than messy high-dimensional data. Numerical safeguards are limited to log-domain density calculations and diagonal covariance regularization.

These limitations are useful opportunities for extension. A future version could compare covariance parameterizations, add repeated-trial benchmark tables, implement information criteria, and include real datasets where model misspecification is more apparent.

## 10. Conclusion

Gaussian Mixture Models connect several core ideas in mathematical machine learning: density geometry, latent variables, Bayesian posterior probabilities, iterative optimization, and probabilistic classification. This project presents those ideas as a compact educational sequence with reproducible notebooks and reusable source code. The result is a foundation that can be extended toward model selection, Bayesian mixtures, anomaly detection, and higher-dimensional applications.