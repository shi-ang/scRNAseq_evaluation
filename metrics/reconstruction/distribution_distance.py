import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import nbinom, poisson

_EPS = 1e-12


def mmd_pert(
        x_obs,
        x_pred,
        kernel='rbf',
        **kernel_params,
):
    """
    Calculate Maximum Mean Discrepancy (MMD^2) between the distributions of observed and predicted values,
    for one perturbation.

    Arguments:
        x_obs: observed post-perturbation profile. Shape: (n_cells, n_genes)
        x_pred: predicted post-perturbation profile. Shape: (n_cells, n_genes)
        kernel: Kernel type to use in MMD calculation. Default is 'rbf'.
            choices are 'rbf', 'linear', 'poly', 'sigmoid'.
        **kernel_params: optional kernel parameters passed to sklearn.metrics.pairwise.pairwise_kernels
            e.g., for rbf/sigmoid: gamma=...
                  for poly: degree=..., gamma=..., coef0=...
                  for sigmoid: gamma=..., coef0=...

    Returns:
        MMD^2 as a float (unbiased estimator when possible; falls back to biased if sample size < 2).
    """
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)

    if x_obs.shape[1] != x_pred.shape[1]:
        raise ValueError(f"x_obs and x_pred must have the same number of genes (features). "
                         f"Got {x_obs.shape[1]} and {x_pred.shape[1]}.")

    m, n = x_obs.shape[0], x_pred.shape[0]

    # Compute Gram matrices
    Kxx = pairwise_kernels(x_obs, x_obs, metric=kernel, filter_params=True, **kernel_params)
    Kyy = pairwise_kernels(x_pred, x_pred, metric=kernel, filter_params=True, **kernel_params)
    Kxy = pairwise_kernels(x_obs, x_pred, metric=kernel, filter_params=True, **kernel_params)
    # Unbiased MMD^2 estimator excludes diagonals of Kxx and Kyy
    if m >= 2 and n >= 2:
        xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        xy = Kxy.mean()
        mmd2 = xx + yy - 2.0 * xy
    else:
        # Fallback to biased estimator if too few samples
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(mmd2)


def parametric_dist(
        x_obs: np.ndarray,
        x_pred: np.ndarray,
        parametric_form: str = "NB",
        dist_type: str = "JS-divergence",
) -> float:
    """
    Calculate distribution distance between observed and predicted values:
      1) fit parametric distributions to both observed and predicted data (per gene),
      2) compute a distance between the two fitted distributions (per gene),
      3) return the mean distance across genes.

    Arguments:
        x_obs: observed post-perturbation profile. Shape: (n_cells, n_genes)
        x_pred: predicted post-perturbation profile. Shape: (n_cells, n_genes)
        parametric_form: "NB" (Negative Binomial), "Poisson", "ZINB" (Zero-Inflated NB).
            NB/ZINB are fit in the common (mu, theta) parameterization:
                Var = mu + mu^2 / theta
        dist_type: "JS-divergence" or "Wasserstein" (1-Wasserstein on counts).
    """
    # ----------------------------
    # Helpers: fitting + pmf + ppf
    # ----------------------------
    q_tail = 1.0 - 1e-8          # truncation quantile
    kmax_cap = 10_000            # safety cap to avoid huge loops

    def _fit_poisson(mu: float):
        lam = max(float(mu), 0.0)
        return {"lam": lam}

    def _fit_nb(mu: float, var: float):
        mu = max(float(mu), 0.0)
        var = max(float(var), 0.0)
        # Method-of-moments for theta: var = mu + mu^2/theta  => theta = mu^2/(var-mu)
        denom = max(var - mu, 0.0)
        if mu <= _EPS:
            theta = 1e8
        elif denom <= 1e-8 * max(mu, 1.0):  # near-Poisson
            theta = 1e8
        else:
            theta = (mu * mu) / max(denom, _EPS)
        theta = float(np.clip(theta, 1e-8, 1e12))
        return {"mu": mu, "theta": theta}

    def _nbinom_from_mu_theta(mu: float, theta: float):
        # SciPy nbinom(n, p): mean = n*(1-p)/p
        n = float(theta)
        p = n / (n + float(mu) + _EPS)
        p = float(np.clip(p, _EPS, 1.0 - _EPS))
        n = float(max(n, _EPS))
        return n, p

    def _fit_zinb(mu: float, var: float, p0: float):
        # crude but stable: fit NB by moments, then set pi from "excess zeros"
        nb = _fit_nb(mu, var)
        n, p = _nbinom_from_mu_theta(nb["mu"], nb["theta"])
        nb_p0 = float(nbinom(n, p).pmf(0))  # = p**n
        p0 = float(np.clip(p0, 0.0, 1.0))
        pi = (p0 - nb_p0) / max(1.0 - nb_p0, _EPS)
        pi = float(np.clip(pi, 0.0, 1.0 - 1e-12))
        return {"mu": nb["mu"], "theta": nb["theta"], "pi": pi}

    def _pmf(params: dict, ks: np.ndarray, form: str) -> np.ndarray:
        if form == "POISSON":
            pmf = poisson(params["lam"]).pmf(ks)
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pmf = nbinom(n, p).pmf(ks)
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            pmf = (1.0 - pi) * nbinom(n, p).pmf(ks)
            pmf[0] += pi
        else:
            raise ValueError(f"Unknown parametric_form: {form}")
        pmf = np.asarray(pmf, dtype=float)
        s = pmf.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to a degenerate-at-zero distribution
            pmf = np.zeros_like(pmf)
            pmf[0] = 1.0
            return pmf
        return pmf / s

    def _ppf(params: dict, q: float, form: str) -> float:
        q = float(np.clip(q, _EPS, 1.0 - _EPS))
        if form == "POISSON":
            return float(poisson(params["lam"]).ppf(q))
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            return float(nbinom(n, p).ppf(q))
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            nb_dist = nbinom(n, p)

            p0 = pi + (1.0 - pi) * float(nb_dist.cdf(0))
            if q <= p0:
                return 0.0

            q_nb = (q - pi) / max(1.0 - pi, _EPS)
            q_nb = float(np.clip(q_nb, _EPS, 1.0 - _EPS))
            return float(nb_dist.ppf(q_nb))
        else:
            raise ValueError(f"Unknown parametric_form: {form}")

    # ----------------------------
    # Helpers: distances on pmfs
    # ----------------------------
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        # KL(p||m) and KL(q||m)
        kl_pm = np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)))
        kl_qm = np.sum(q * (np.log(q + _EPS) - np.log(m + _EPS)))
        return float(0.5 * (kl_pm + kl_qm))

    def _wasserstein_1(p: np.ndarray, q: np.ndarray) -> float:
        # Exact 1-Wasserstein on integers with unit spacing: sum_k |CDF_p(k)-CDF_q(k)|
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)))

    # ----------------------------
    # Main
    # ----------------------------
    x_obs = np.asarray(x_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    if x_obs.shape != x_pred.shape:
        raise ValueError(f"x_obs and x_pred must have the same shape, got {x_obs.shape} vs {x_pred.shape}")

    # Count models: clip negatives
    x_obs = np.clip(x_obs, 0.0, None)
    x_pred = np.clip(x_pred, 0.0, None)

    form = parametric_form.strip().upper()
    dist = dist_type.strip().upper()

    n_cells, n_genes = x_obs.shape
    mu_obs = x_obs.mean(axis=0)
    mu_pred = x_pred.mean(axis=0)
    var_obs = x_obs.var(axis=0, ddof=1) if n_cells > 1 else np.zeros(n_genes)
    var_pred = x_pred.var(axis=0, ddof=1) if n_cells > 1 else np.zeros(n_genes)
    p0_obs = (x_obs == 0).mean(axis=0)
    p0_pred = (x_pred == 0).mean(axis=0)

    dists = []

    for g in range(n_genes):
        if form == "POISSON":
            p_params = _fit_poisson(mu_obs[g])
            q_params = _fit_poisson(mu_pred[g])
        elif form == "NB":
            p_params = _fit_nb(mu_obs[g], var_obs[g])
            q_params = _fit_nb(mu_pred[g], var_pred[g])
        elif form == "ZINB":
            p_params = _fit_zinb(mu_obs[g], var_obs[g], p0_obs[g])
            q_params = _fit_zinb(mu_pred[g], var_pred[g], p0_pred[g])
        else:
            raise ValueError('parametric_form must be one of {"NB","Poisson","ZINB"}')

        # choose truncation support
        data_max = int(max(np.max(x_obs[:, g]), np.max(x_pred[:, g])))
        k1 = _ppf(p_params, q_tail, form)
        k2 = _ppf(q_params, q_tail, form)
        kmax = int(min(max(data_max, k1, k2), kmax_cap))
        ks = np.arange(kmax + 1, dtype=int)

        p_pmf = _pmf(p_params, ks, form)
        q_pmf = _pmf(q_params, ks, form)

        if dist in {"JS-DIVERGENCE", "JS"}:
            d = _js_divergence(p_pmf, q_pmf)
        elif dist in {"WASSERSTEIN", "WASSERSTEIN-1", "W1"}:
            d = _wasserstein_1(p_pmf, q_pmf)
        else:
            raise ValueError('dist_type must be "JS-divergence" or "Wasserstein"')

        if np.isfinite(d):
            dists.append(d)

    return float(np.mean(dists)) if len(dists) > 0 else float("nan")
